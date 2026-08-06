"""Phase 2 — detect similar past experiments before creating or launching."""

from __future__ import annotations

import re
from typing import Any

from .experiment_store import ExperimentStore, get_experiment
from .finding_extraction import _finding_from_change, _goal_finding


def _change_signature(change: dict[str, Any]) -> str:
    spec = _finding_from_change(change)
    if spec:
        return str(spec["finding_key"])
    text = str(change.get("text") or change.get("key") or "").strip().lower()
    return re.sub(r"\s+", "_", text)[:80] or "unknown"


def _goal_tokens(goal: str) -> set[str]:
    stop = {"reduce", "failures", "failure", "the", "a", "an", "to", "from", "improve", "overall", "grade"}
    tokens = set(re.findall(r"[a-z0-9]+", str(goal or "").lower()))
    return {t for t in tokens if t not in stop and len(t) > 2}


def _experiment_signatures(experiment: dict[str, Any]) -> tuple[set[str], set[str]]:
    changes = set(_change_signature(c) for c in (experiment.get("accepted_changes") or []))
    goal_spec = _goal_finding(str(experiment.get("goal") or ""))
    if goal_spec:
        changes.add(str(goal_spec["finding_key"]))
    goals = _goal_tokens(str(experiment.get("goal") or ""))
    return changes, goals


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _experiment_outcome(experiment: dict[str, Any]) -> dict[str, Any]:
    results = experiment.get("results") or {}
    pf_b = _num(results.get("profit_factor_before"))
    pf_a = _num(results.get("profit_factor_after"))
    wr_b = _num(results.get("win_rate_before_pct"))
    wr_a = _num(results.get("win_rate_after_pct"))
    if pf_b is None or pf_a is None:
        return {"outcome": "unknown", "pf_change": None, "win_rate_change": None}
    pf_change = pf_a - pf_b
    wr_change = (wr_a - wr_b) if wr_b is not None and wr_a is not None else None
    if pf_change > 0.02 or (wr_change is not None and wr_change > 1.0):
        return {"outcome": "improved", "pf_change": round(pf_change, 3), "win_rate_change": wr_change}
    if pf_change < -0.02 or (wr_change is not None and wr_change < -1.0):
        return {"outcome": "no_improvement", "pf_change": round(pf_change, 3), "win_rate_change": wr_change}
    return {"outcome": "mixed", "pf_change": round(pf_change, 3), "win_rate_change": wr_change}


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _similarity_score(
    proposed_changes: set[str],
    proposed_goals: set[str],
    proposed_model: str | None,
    past: dict[str, Any],
) -> tuple[float, list[str]]:
    past_changes, past_goals = _experiment_signatures(past)
    change_sim = _jaccard(proposed_changes, past_changes)
    goal_sim = _jaccard(proposed_goals, past_goals) if (proposed_goals or past_goals) else 0.0
    model_bonus = 0.1 if proposed_model and proposed_model == (past.get("provenance") or {}).get("model_id") else 0.0
    score = min(1.0, change_sim * 0.72 + goal_sim * 0.18 + model_bonus)
    shared = sorted(proposed_changes & past_changes)
    return round(score * 100, 1), shared


def _verdict(top_pct: float) -> str:
    if top_pct >= 80:
        return "very_similar"
    if top_pct >= 45:
        return "similar"
    return "novel"


def _recommendation_text(top: dict[str, Any] | None, verdict: str) -> str:
    if verdict == "novel":
        return "Novel experiment — low overlap with past work. Recommended to proceed."
    if not top:
        return ""
    num = top.get("experiment_number")
    pct = top.get("similarity_pct")
    outcome = top.get("outcome")
    if verdict == "very_similar" and outcome == "no_improvement":
        return (
            f"Very similar to Experiment #{num} ({pct}% overlap) which did not improve results. "
            "Consider changing the hypothesis before proceeding."
        )
    if verdict == "very_similar" and outcome == "improved":
        return f"Similar to Experiment #{num} ({pct}% overlap) which improved PF. May be worth repeating."
    if outcome == "no_improvement":
        return f"Similar to Experiment #{num} ({pct}% overlap) — prior result: no improvement."
    return f"Partial overlap with Experiment #{num} ({pct}%). Review before proceeding."


def check_similar_experiments(
    data_dir: str,
    *,
    accepted_items: list[dict[str, Any]],
    goal: str | None = None,
    model_id: str | None = None,
    exclude_experiment_id: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    proposed_changes = set(_change_signature(item) for item in accepted_items)
    goal_spec = _goal_finding(str(goal or ""))
    if goal_spec:
        proposed_changes.add(str(goal_spec["finding_key"]))
    proposed_goals = _goal_tokens(str(goal or ""))

    if not proposed_changes and not proposed_goals:
        return {"ok": True, "verdict": "novel", "top_similarity_pct": 0, "matches": [], "recommendation": ""}

    with ExperimentStore(data_dir) as store:
        rows = store.list_experiments(limit=200)

    matches: list[dict[str, Any]] = []
    for row in rows:
        exp_id = str(row.get("experiment_id") or "")
        if exclude_experiment_id and exp_id == exclude_experiment_id:
            continue
        if row.get("status") not in ("completed", "launched", "cancelled"):
            continue
        past = get_experiment(data_dir, exp_id)
        if not past:
            continue
        pct, shared = _similarity_score(proposed_changes, proposed_goals, model_id, past)
        if pct < 15:
            continue
        outcome = _experiment_outcome(past)
        matches.append({
            "experiment_id": exp_id,
            "experiment_number": past.get("experiment_number"),
            "similarity_pct": pct,
            "goal": past.get("goal"),
            "status": past.get("status"),
            "shared_changes": shared,
            "outcome": outcome["outcome"],
            "pf_change": outcome.get("pf_change"),
            "win_rate_change": outcome.get("win_rate_change"),
            "model_id": (past.get("provenance") or {}).get("model_id"),
        })

    matches.sort(key=lambda m: m["similarity_pct"], reverse=True)
    matches = matches[: max(1, limit)]
    top_pct = matches[0]["similarity_pct"] if matches else 0.0
    verdict = _verdict(top_pct)
    top = matches[0] if matches else None
    return {
        "ok": True,
        "verdict": verdict,
        "top_similarity_pct": top_pct,
        "matches": matches,
        "recommendation": _recommendation_text(top, verdict),
        "should_warn": verdict == "very_similar" and bool(top) and top.get("outcome") == "no_improvement",
    }


def check_experiment_before_create(
    data_dir: str,
    report: dict[str, Any],
    *,
    accepted_items: list[dict[str, Any]],
    goal: str | None = None,
) -> dict[str, Any]:
    exec_sum = report.get("executive_summary") or {}
    return check_similar_experiments(
        data_dir,
        accepted_items=accepted_items,
        goal=goal,
        model_id=exec_sum.get("model_id"),
    )


def check_experiment_before_launch(data_dir: str, experiment_id: str) -> dict[str, Any]:
    doc = get_experiment(data_dir, experiment_id)
    if not doc:
        return {"ok": False, "error": "experiment not found"}
    prov = doc.get("provenance") or {}
    items = []
    for ch in doc.get("accepted_changes") or []:
        items.append({
            "text": ch.get("text"),
            "target": ch.get("target"),
            "filters": ch.get("filters"),
            "feature_hints": ch.get("feature_hints"),
        })
    return check_similar_experiments(
        data_dir,
        accepted_items=items,
        goal=doc.get("goal"),
        model_id=prov.get("model_id"),
        exclude_experiment_id=experiment_id,
    )
