"""Experiment Planner — bridge between Research Report and execution."""

from __future__ import annotations

from typing import Any

from .experiment_recommendations import build_planner_items_from_report, infer_experiment_goal
from .experiment_store import get_experiment, list_experiments, save_experiment


def build_experiment_planner_view(report: dict[str, Any]) -> dict[str, Any]:
    if not report.get("ok"):
        return report
    exec_sum = report.get("executive_summary") or {}
    baseline = report.get("baseline_metrics") or {}
    return {
        "ok": True,
        "research_report_id": report.get("report_id"),
        "prediction_run_id": report.get("prediction_run_id"),
        "strategy_run_id": report.get("strategy_run_id"),
        "model_id": exec_sum.get("model_id"),
        "strategy_label": exec_sum.get("strategy"),
        "baseline_grade": exec_sum.get("overall_grade"),
        "baseline_pf": baseline.get("profit_factor"),
        "baseline_win_rate_pct": baseline.get("win_rate_pct"),
        "suggested_goal": infer_experiment_goal(report),
        "items": build_planner_items_from_report(report),
        "estimated_improvement": (report.get("action_plan") or {}).get("estimated_improvement"),
    }


def create_experiment_from_report(
    data_dir: str,
    report: dict[str, Any],
    *,
    accepted_items: list[dict[str, Any]],
    goal: str | None = None,
) -> dict[str, Any]:
    if not report.get("ok"):
        return report
    if not accepted_items:
        return {"ok": False, "error": "Select at least one recommendation"}

    exec_sum = report.get("executive_summary") or {}
    baseline = report.get("baseline_metrics") or {}
    est = (report.get("action_plan") or {}).get("estimated_improvement") or {}

    changes = []
    for item in accepted_items:
        changes.append({
            "text": item.get("text"),
            "target": item.get("target"),
            "target_label": item.get("target_label"),
            "filters": item.get("filters") or {},
            "feature_hints": item.get("feature_hints") or [],
            "accepted": True,
        })

    experiment = {
        "goal": goal or infer_experiment_goal(report),
        "status": "pending",
        "provenance": {
            "research_report_id": report.get("report_id"),
            "prediction_run_id": report.get("prediction_run_id"),
            "strategy_run_id": report.get("strategy_run_id"),
            "model_id": exec_sum.get("model_id"),
            "strategy_label": exec_sum.get("strategy"),
            "baseline_grade": exec_sum.get("overall_grade"),
            "baseline_pf": baseline.get("profit_factor"),
            "baseline_win_rate_pct": baseline.get("win_rate_pct"),
        },
        "accepted_changes": changes,
        "expected_improvement": est,
        "results": None,
    }
    saved = save_experiment(data_dir, experiment)
    return {"ok": True, "experiment": saved}


def launch_experiment(data_dir: str, experiment_id: str) -> dict[str, Any]:
    doc = get_experiment(data_dir, experiment_id)
    if not doc:
        return {"ok": False, "error": "experiment not found"}
    if doc.get("status") not in ("pending",):
        return {"ok": False, "error": f"cannot launch experiment in status {doc.get('status')}"}
    from .experiment_store import _utc_now

    doc["status"] = "launched"
    doc["launched_at"] = _utc_now()
    saved = save_experiment(data_dir, doc)
    return {"ok": True, "experiment": saved}


def complete_experiment(
    data_dir: str,
    experiment_id: str,
    *,
    results: dict[str, Any],
) -> dict[str, Any]:
    doc = get_experiment(data_dir, experiment_id)
    if not doc:
        return {"ok": False, "error": "experiment not found"}
    from .experiment_store import _utc_now

    doc["status"] = "completed"
    doc["completed_at"] = _utc_now()
    doc["results"] = dict(results)
    saved = save_experiment(data_dir, doc)

    trade_count = results.get("trade_count")
    if trade_count is None:
        report_id = (doc.get("provenance") or {}).get("research_report_id")
        if report_id:
            from .research_report_store import get_research_report

            report = get_research_report(data_dir, str(report_id))
            if report:
                trade_count = (report.get("executive_summary") or {}).get("trade_count")

    from .finding_extraction import extract_findings_from_experiment

    knowledge = extract_findings_from_experiment(data_dir, saved, trade_count=trade_count)
    return {"ok": True, "experiment": saved, "knowledge_extraction": knowledge}


def list_experiment_board(data_dir: str, *, limit: int = 50) -> dict[str, Any]:
    rows = list_experiments(data_dir, limit=limit)
    return {"ok": True, "experiments": rows}
