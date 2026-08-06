"""Extract evidence-backed findings from completed experiments."""

from __future__ import annotations

from typing import Any

from .knowledge_store import KnowledgeStore


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _finding_from_change(change: dict[str, Any]) -> dict[str, Any] | None:
    text = str(change.get("text") or "").strip()
    low = text.lower()
    target = str(change.get("target") or "")
    filters = change.get("filters") or {}
    hints = change.get("feature_hints") or []

    if "premium" in low and ("below" in low or "avoid" in low or "<" in low):
        threshold = int(filters.get("min_premium") or 25)
        return {
            "finding_key": f"premium_below_{threshold}_poor",
            "finding": f"Premium < ₹{threshold} performs poorly",
            "category": "strategy",
        }
    if "confidence" in low:
        threshold = int(filters.get("min_confidence") or 70)
        return {
            "finding_key": f"low_confidence_below_{threshold}_poor",
            "finding": f"Confidence < {threshold}% entries underperform",
            "category": "strategy",
        }
    if "stop" in low and ("7" in low or "%" in low):
        return {
            "finding_key": "stop_7pct_improves_outcomes",
            "finding": "7% stop loss improves trade outcomes",
            "category": "strategy",
        }
    if "theta" in low and "filter" in low:
        return {
            "finding_key": "high_theta_trades_underperform",
            "finding": "High theta trades underperform",
            "category": "strategy",
        }
    if "range" in low:
        return {
            "finding_key": "range_regime_underperform",
            "finding": "Range regime entries underperform",
            "category": "strategy",
        }
    if "theta" in low and ("feature" in low or "retrain" in low or target == "feature_registry"):
        return {
            "finding_key": "theta_feature_improves_model",
            "finding": "Theta features improve model accuracy",
            "category": "feature",
        }
    if "iv" in low and ("feature" in low or "expansion" in low):
        return {
            "finding_key": "iv_feature_improves_model",
            "finding": "IV expansion features improve model accuracy",
            "category": "feature",
        }
    if target == "model_builder" or "retrain" in low:
        return {
            "finding_key": "model_retrain_improves_performance",
            "finding": "Model retrain improves walk-forward performance",
            "category": "model",
        }
    if target == "hyperparameter_optimization":
        return {
            "finding_key": "hpo_improves_model",
            "finding": "Hyperparameter optimization improves model performance",
            "category": "model",
        }
    if hints:
        key = "_".join(sorted(hints))
        return {
            "finding_key": f"feature_{key}_helps",
            "finding": f"{' / '.join(hints)} features improve outcomes",
            "category": "feature",
        }
    return None


def _goal_finding(goal: str) -> dict[str, Any] | None:
    low = str(goal or "").lower()
    if "theta" in low:
        return {
            "finding_key": "theta_decay_is_major_failure_mode",
            "finding": "Theta decay is a major failure mode",
            "category": "market",
        }
    if "premium" in low:
        return {
            "finding_key": "low_premium_is_major_failure_mode",
            "finding": "Low premium entries are a major failure mode",
            "category": "market",
        }
    if "direction" in low or "prediction" in low:
        return {
            "finding_key": "wrong_direction_is_major_failure_mode",
            "finding": "Wrong prediction direction is a major failure mode",
            "category": "model",
        }
    return None


def _evidence_quality(trade_count: int | None, pf_change: float | None) -> str:
    tc = trade_count or 0
    pf = abs(pf_change or 0)
    if tc >= 1000 and pf >= 0.1:
        return "strong"
    if tc >= 200 or pf >= 0.05:
        return "moderate"
    return "weak"


def _supports_finding(pf_change: float | None, win_rate_change: float | None) -> bool:
    pf = pf_change or 0
    wr = win_rate_change or 0
    return pf > 0.02 or wr > 1.0


def extract_findings_from_experiment(
    data_dir: str,
    experiment: dict[str, Any],
    *,
    trade_count: int | None = None,
    campaign_id: str | None = None,
    program_id: str | None = None,
) -> dict[str, Any]:
    """Run after experiment completion — upsert findings and append evidence."""
    results = experiment.get("results") or {}
    prov = experiment.get("provenance") or {}
    pf_before = _num(results.get("profit_factor_before")) or _num(prov.get("baseline_pf"))
    pf_after = _num(results.get("profit_factor_after"))
    wr_before = _num(results.get("win_rate_before_pct")) or _num(prov.get("baseline_win_rate_pct"))
    wr_after = _num(results.get("win_rate_after_pct"))
    pf_change = (pf_after - pf_before) if pf_before is not None and pf_after is not None else None
    wr_change = (wr_after - wr_before) if wr_before is not None and wr_after is not None else None
    supports = _supports_finding(pf_change, wr_change)

    specs: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for change in experiment.get("accepted_changes") or []:
        spec = _finding_from_change(change)
        if spec and spec["finding_key"] not in seen_keys:
            seen_keys.add(spec["finding_key"])
            specs.append(spec)

    goal_spec = _goal_finding(str(experiment.get("goal") or ""))
    if goal_spec and goal_spec["finding_key"] not in seen_keys:
        specs.append(goal_spec)

    if not specs:
        return {"ok": True, "findings_updated": 0, "note": "No extractable findings"}

    exp_id = experiment.get("experiment_id")
    exp_num = experiment.get("experiment_number")
    report_id = prov.get("research_report_id")
    pred_id = prov.get("prediction_run_id")
    strat_id = prov.get("strategy_run_id")
    model_id = prov.get("model_id")
    strategy_label = prov.get("strategy_label")
    campaign_id = campaign_id or prov.get("campaign_id")
    program_id = program_id or prov.get("program_id")

    updated: list[dict[str, Any]] = []
    with KnowledgeStore(data_dir) as store:
        for spec in specs:
            finding = store.upsert_finding(
                finding_key=spec["finding_key"],
                finding=spec["finding"],
                category=spec["category"],
                metadata={"source": "experiment_extraction"},
            )
            fid = finding.get("finding_id")
            if not fid:
                continue

            eq = _evidence_quality(trade_count, pf_change)
            store.add_evidence(
                fid,
                {
                    "experiment_id": exp_id,
                    "experiment_number": exp_num,
                    "research_report_id": report_id,
                    "prediction_run_id": pred_id,
                    "strategy_run_id": strat_id,
                    "model_id": model_id,
                    "campaign_id": campaign_id,
                    "program_id": program_id,
                    "trade_count": trade_count,
                    "pf_change": round(pf_change, 4) if pf_change is not None else None,
                    "win_rate_change": round(wr_change, 2) if wr_change is not None else None,
                    "supports_finding": supports,
                    "evidence_quality": eq,
                    "notes": f"Experiment #{exp_num} — PF Δ {pf_change:+.2f}" if pf_change is not None else None,
                },
            )

            if report_id:
                store.add_link(fid, link_type="research_report", link_ref=str(report_id))
            if pred_id:
                store.add_link(fid, link_type="prediction_run", link_ref=str(pred_id))
            if strat_id:
                store.add_link(fid, link_type="strategy_run", link_ref=str(strat_id), link_label=strategy_label)
            if model_id:
                store.add_link(fid, link_type="model", link_ref=str(model_id))
            template_id = prov.get("template_id")
            if template_id:
                store.add_link(
                    fid,
                    link_type="experiment_template",
                    link_ref=str(template_id),
                    link_label=f"Template #{prov.get('template_number') or ''}".strip(),
                )
            if campaign_id:
                store.add_link(fid, link_type="research_campaign", link_ref=str(campaign_id))
            if program_id:
                store.add_link(fid, link_type="research_program", link_ref=str(program_id))
            job_id = prov.get("job_id") or exp_id
            if job_id:
                store.add_link(
                    fid,
                    link_type="experiment_job",
                    link_ref=str(job_id),
                    link_label=f"Job #{prov.get('job_number') or exp_num or ''}".strip(),
                )
            for change in experiment.get("accepted_changes") or []:
                for hint in change.get("feature_hints") or []:
                    store.add_link(fid, link_type="feature", link_ref=str(hint), link_label=str(hint))

            refreshed = store.get_finding(fid)
            if refreshed:
                updated.append({
                    "finding_id": fid,
                    "finding": refreshed.get("finding"),
                    "status": refreshed.get("status"),
                    "confidence": refreshed.get("confidence"),
                    "evidence_count": refreshed.get("evidence_count"),
                })

    return {"ok": True, "findings_updated": len(updated), "findings": updated}


def extract_findings_from_job(
    data_dir: str,
    *,
    template: dict[str, Any],
    job: dict[str, Any],
    comparison: dict[str, Any],
    trade_count: int | None = None,
    campaign_id: str | None = None,
    program_id: str | None = None,
) -> dict[str, Any]:
    """Run after experiment job completion — reuse legacy finding extraction."""
    pseudo = {
        "experiment_id": job.get("job_id"),
        "experiment_number": job.get("job_number"),
        "goal": template.get("goal"),
        "accepted_changes": template.get("accepted_changes") or [],
        "provenance": {
            "research_report_id": (job.get("outputs") or {}).get("research_report_id") or template.get("research_report_id"),
            "prediction_run_id": (job.get("outputs") or {}).get("prediction_run_id") or template.get("prediction_run_id"),
            "strategy_run_id": (job.get("outputs") or {}).get("strategy_run_id") or template.get("strategy_run_id"),
            "model_id": template.get("model_id"),
            "strategy_label": template.get("strategy_label"),
            "baseline_pf": comparison.get("baseline_pf"),
            "baseline_win_rate_pct": comparison.get("baseline_win_rate_pct"),
            "template_id": template.get("template_id"),
            "template_number": template.get("template_number"),
            "job_id": job.get("job_id"),
            "job_number": job.get("job_number"),
            "campaign_id": campaign_id or template.get("campaign_id") or job.get("campaign_id"),
            "program_id": program_id or template.get("program_id") or job.get("program_id"),
        },
        "results": {
            "profit_factor_before": comparison.get("baseline_pf"),
            "profit_factor_after": comparison.get("after_pf"),
            "win_rate_before_pct": comparison.get("baseline_win_rate_pct"),
            "win_rate_after_pct": comparison.get("after_win_rate_pct"),
            "grade": comparison.get("after_grade"),
        },
    }
    return extract_findings_from_experiment(
        data_dir,
        pseudo,
        trade_count=trade_count,
        campaign_id=campaign_id or template.get("campaign_id") or job.get("campaign_id"),
        program_id=program_id or template.get("program_id") or job.get("program_id"),
    )
