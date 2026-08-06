"""Automatic post-job pipeline — zero manual work after Run."""

from __future__ import annotations

from typing import Any


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _pct_delta(before: Any, after: Any) -> float | None:
    b, a = _num(before), _num(after)
    if b is None or a is None or b == 0:
        return None
    return round((a - b) / abs(b) * 100.0, 2)


def collect_job_outputs(
    *,
    template: dict[str, Any],
    job: dict[str, Any],
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Gather all artifact IDs and metrics the platform already produced."""
    outputs = dict(job.get("outputs") or {})
    clone_info = ctx.get("clone") or {}
    model_prep = ctx.get("model_prep") or {}
    for key, val in (
        ("prediction_run_id", ctx.get("prediction_run_id")),
        ("strategy_run_id", ctx.get("strategy_run_id")),
        ("research_report_id", ctx.get("research_report_id")),
        ("model_name", ctx.get("model_name")),
        ("strategy_version_id", ctx.get("strategy_version_id")),
        ("strategy_version_label", clone_info.get("version_label")),
        ("training_mode", model_prep.get("mode")),
        ("phase", ctx.get("job_phase")),
    ):
        if val and not outputs.get(key):
            outputs[key] = val
    baseline = template.get("baseline") or {}
    after_metrics = ctx.get("metrics") or {}
    training = ctx.get("training_result") or {}
    wf = training.get("walk_forward") or {}
    train_metrics = training.get("metrics") or {}
    val_metrics = train_metrics.get("validation") or {}
    test_metrics = train_metrics.get("test") or {}

    return {
        "template_id": template.get("template_id"),
        "template_number": template.get("template_number"),
        "job_id": job.get("job_id"),
        "job_number": job.get("job_number"),
        "phase": outputs.get("phase"),
        "baseline_prediction_run_id": outputs.get("baseline_prediction_run_id") or template.get("prediction_run_id"),
        "prediction_run_id": outputs.get("prediction_run_id") or ctx.get("prediction_run_id"),
        "strategy_run_id": outputs.get("strategy_run_id") or ctx.get("strategy_run_id"),
        "research_report_id": outputs.get("research_report_id") or ctx.get("research_report_id"),
        "model_name": outputs.get("model_name") or ctx.get("model_name"),
        "model_id": template.get("model_id"),
        "strategy_version_id": outputs.get("strategy_version_id") or ctx.get("strategy_version_id"),
        "strategy_version_label": outputs.get("strategy_version_label"),
        "training_mode": outputs.get("training_mode"),
        "strategy": {
            "profit_factor": after_metrics.get("profit_factor"),
            "win_rate_pct": after_metrics.get("win_rate_pct"),
            "trade_count": after_metrics.get("trade_count"),
            "max_drawdown": after_metrics.get("max_drawdown"),
            "expectancy": after_metrics.get("expectancy"),
        },
        "baseline_strategy": {
            "profit_factor": baseline.get("profit_factor"),
            "win_rate_pct": baseline.get("win_rate_pct"),
            "trade_count": baseline.get("trade_count"),
            "grade": baseline.get("grade"),
        },
        "model_metrics": {
            "mae": wf.get("mean_mae") or val_metrics.get("mae"),
            "rmse": wf.get("mean_rmse") or val_metrics.get("rmse"),
            "directional_accuracy_pct": wf.get("mean_directional_accuracy_pct") or val_metrics.get("directional_accuracy_pct"),
            "test_mae": test_metrics.get("mae"),
            "test_rmse": test_metrics.get("rmse"),
        },
    }


def compare_with_baseline(collected: dict[str, Any]) -> dict[str, Any]:
    base = collected.get("baseline_strategy") or {}
    after = collected.get("strategy") or {}
    model = collected.get("model_metrics") or {}

    pf_b, pf_a = _num(base.get("profit_factor")), _num(after.get("profit_factor"))
    wr_b, wr_a = _num(base.get("win_rate_pct")), _num(after.get("win_rate_pct"))
    tc_b, tc_a = _num(base.get("trade_count")), _num(after.get("trade_count"))

    return {
        "baseline_pf": pf_b,
        "after_pf": pf_a,
        "pf_delta": round(pf_a - pf_b, 4) if pf_b is not None and pf_a is not None else None,
        "pf_delta_pct": _pct_delta(pf_b, pf_a),
        "baseline_win_rate_pct": wr_b,
        "after_win_rate_pct": wr_a,
        "win_rate_delta": round(wr_a - wr_b, 2) if wr_b is not None and wr_a is not None else None,
        "win_rate_delta_pct": _pct_delta(wr_b, wr_a),
        "baseline_trade_count": int(tc_b) if tc_b is not None else None,
        "after_trade_count": int(tc_a) if tc_a is not None else None,
        "trade_count_delta": int(tc_a - tc_b) if tc_b is not None and tc_a is not None else None,
        "trade_count_delta_pct": _pct_delta(tc_b, tc_a),
        "baseline_grade": base.get("grade"),
        "after_grade": collected.get("after_grade"),
        "after_mae": model.get("mae"),
        "after_rmse": model.get("rmse"),
        "after_directional_accuracy_pct": model.get("directional_accuracy_pct"),
    }


def _impact_label(
    delta: float | None,
    threshold: float,
    *,
    pct: bool = False,
) -> str:
    if delta is None:
        return "Unknown"
    value = abs(delta)
    if pct:
        if value < threshold:
            return "No Change"
    elif value < threshold:
        return "No Change"
    return "Improved" if delta > 0 else "Declined"


def build_trading_impact(
    comparison: dict[str, Any],
    *,
    collected: dict[str, Any],
    template: dict[str, Any],
    verdict: dict[str, Any],
    outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Human-readable trading + prediction interpretation for completed jobs."""
    outputs = outputs or {}
    pf_delta = _num(comparison.get("pf_delta"))
    wr_delta = _num(comparison.get("win_rate_delta"))
    trade_delta_pct = _num(comparison.get("trade_count_delta_pct"))

    pf_label = _impact_label(pf_delta, 0.05)
    wr_label = _impact_label(wr_delta, 1.0)
    trades_label = _impact_label(trade_delta_pct, 5.0, pct=True)

    phase = str(outputs.get("phase") or collected.get("phase") or "")
    baseline_pred = str(
        outputs.get("baseline_prediction_run_id")
        or collected.get("baseline_prediction_run_id")
        or template.get("prediction_run_id")
        or ""
    )
    after_pred = str(outputs.get("prediction_run_id") or collected.get("prediction_run_id") or "")
    had_training = phase in ("C", "B+C") or bool(outputs.get("training_mode"))
    retrained = had_training and bool(after_pred) and after_pred != baseline_pred

    da = _num(comparison.get("after_directional_accuracy_pct"))
    if not retrained and not had_training:
        prediction = "Unchanged"
    elif da is not None and da >= 52:
        prediction = "Improved"
    elif comparison.get("after_mae") is not None or comparison.get("after_rmse") is not None:
        prediction = "Improved"
    elif retrained or had_training:
        prediction = "Retrained"
    else:
        prediction = "Unchanged"

    verdict_label = verdict.get("verdict") or "Neutral"
    routing = template.get("routing") or {}
    had_strategy = bool(routing.get("strategy_changes"))
    had_model = bool(
        routing.get("feature_changes")
        or routing.get("model_changes")
        or routing.get("optimization_changes")
    )

    conclusion = ""
    if retrained and pf_label == "No Change" and trades_label == "No Change":
        conclusion = "Retraining produced no trading benefit."
    elif retrained and verdict_label == "Regression":
        conclusion = "Retraining did not offset strategy-level regression in this bundle."
    elif had_strategy and not had_model and pf_label == "Declined":
        conclusion = "Strategy filter changes reduced performance."
    elif had_strategy and trades_label == "Declined" and pf_label != "Improved":
        conclusion = "Filters reduced trade count without improving profit factor."
    elif verdict_label == "Improvement":
        conclusion = "Changes improved trading performance — consider promotion."
    elif verdict_label == "Regression":
        n_changes = len(template.get("accepted_changes") or [])
        if n_changes > 1:
            conclusion = "Bundled changes hurt performance — isolate single changes next."
        else:
            conclusion = "This change reduced performance — reject or refine the hypothesis."
    elif prediction == "Improved" and pf_label == "No Change":
        conclusion = "Model metrics improved but trading impact was neutral."
    else:
        conclusion = "No meaningful trading impact detected — narrow the hypothesis next."

    return {
        "pf": pf_label,
        "win_rate": wr_label,
        "trades": trades_label,
        "prediction": prediction,
        "conclusion": conclusion,
        "retrained": retrained,
        "had_training": had_training,
        "had_strategy": had_strategy,
    }


def default_job_decision(
    verdict: dict[str, Any],
    *,
    template: dict[str, Any],
    outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Default researcher decision checklist — archive is always valuable."""
    outputs = outputs or {}
    verdict_label = verdict.get("verdict") or "Neutral"
    routing = template.get("routing") or {}
    phase = str(outputs.get("phase") or "")

    promote_strategy = verdict_label == "Improvement" and bool(routing.get("strategy_changes"))
    promote_model = (
        verdict_label == "Improvement"
        and phase in ("C", "B+C")
        and bool(routing.get("feature_changes") or routing.get("model_changes"))
    )
    repeat = verdict_label in ("Regression", "Neutral") and len(template.get("accepted_changes") or []) >= 1

    return {
        "promote_strategy": promote_strategy,
        "promote_model": promote_model,
        "archive_as_evidence": True,
        "repeat_modified_hypothesis": repeat and not promote_strategy and not promote_model,
        "notes": "",
    }


def _change_row_from_planner_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": item.get("text"),
        "target": item.get("target"),
        "target_label": item.get("target_label"),
        "filters": item.get("filters") or {},
        "feature_hints": item.get("feature_hints") or [],
    }


def recommend_follow_up_from_report(
    data_dir: str,
    *,
    template: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any] | None:
    """Pick an untried recommendation from the linked research report."""
    from .experiment_recommendations import build_planner_items_from_report
    from .research_report_store import get_research_report

    outputs = job.get("outputs") or {}
    report_id = str(outputs.get("research_report_id") or "")
    if not report_id:
        return None
    report = get_research_report(data_dir, report_id)
    if not report:
        return None

    tried = {
        str(c.get("text") or "").strip().lower()
        for c in (template.get("accepted_changes") or [])
    }
    items = build_planner_items_from_report(report)
    for item in items:
        text = str(item.get("text") or "").strip()
        if not text or text.lower() in tried:
            continue
        row = _change_row_from_planner_item(item)
        title = text[:64]
        if "premium" in text.lower():
            title = "Isolate Premium Filter"
        return {
            "title": title,
            "goal": f"Isolate: {text}",
            "selection": [row],
            "expected_information_gain": "High",
            "stars": int(item.get("stars") or 4),
            "reason": "Untried recommendation from research report",
            "source": "research_report",
        }
    return None


def recommend_follow_up_experiment(
    data_dir: str,
    *,
    template: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any] | None:
    """Pick the best follow-up experiment for one-click template creation."""
    results = job.get("results") or {}
    comparison = job.get("comparison") or {}
    verdict = results.get("verdict") or {}
    information_gain = results.get("information_gain") or {}

    next_items = results.get("next_experiments") or []
    if not next_items:
        next_items = suggest_next_experiments(
            template,
            comparison=comparison,
            verdict=verdict,
            information_gain=information_gain,
        )
    if next_items:
        pick = next_items[0]
        if pick.get("selection"):
            return pick

    from_report = recommend_follow_up_from_report(data_dir, template=template, job=job)
    if from_report and from_report.get("selection"):
        return from_report

    root = str((results.get("root_cause") or {}).get("most_likely") or "")
    changes = template.get("accepted_changes") or []
    if len(changes) == 1 and "analysis-only" in root.lower():
        ch = changes[0]
        return {
            "title": "Try Next Report Recommendation",
            "goal": "Current filter is analysis-only — test a different isolated hypothesis",
            "selection": [ch],
            "expected_information_gain": "Medium",
            "stars": 3,
            "reason": root,
            "source": "root_cause",
        }

    if verdict.get("verdict") == "Regression" and len(changes) > 1:
        strategy_only = [
            c for c in changes if str(c.get("target")) == "strategy_registry"
        ]
        if strategy_only:
            return {
                "title": "Strategy Filters Only",
                "goal": "Isolate strategy filters on baseline model predictions",
                "selection": strategy_only[:1],
                "expected_information_gain": "Very High",
                "stars": 5,
                "reason": "Bundled regression — isolate strategy effect",
                "source": "verdict",
            }

    return None


def generate_experiment_verdict(
    comparison: dict[str, Any],
    *,
    collected: dict[str, Any],
    template: dict[str, Any],
) -> dict[str, Any]:
    pf_delta = _num(comparison.get("pf_delta"))
    wr_delta = _num(comparison.get("win_rate_delta"))
    trade_delta_pct = _num(comparison.get("trade_count_delta_pct"))
    n_changes = len(template.get("accepted_changes") or [])

    reasons: list[str] = []
    if pf_delta is not None:
        if pf_delta > 0.05:
            reasons.append(f"PF increased by {pf_delta:+.2f}")
        elif pf_delta < -0.05:
            reasons.append(f"PF decreased by {pf_delta:+.2f}")
        else:
            reasons.append("PF largely unchanged")

    if trade_delta_pct is not None and abs(trade_delta_pct) >= 5:
        direction = "reduced" if trade_delta_pct < 0 else "increased"
        reasons.append(f"Trade count {direction} by {abs(trade_delta_pct):.1f}%")

    if wr_delta is not None and abs(wr_delta) >= 1.0:
        reasons.append(f"Win rate changed by {wr_delta:+.1f}%")

    if comparison.get("after_mae") is not None:
        reasons.append("Model error metrics recorded from training")

    if pf_delta is not None and pf_delta > 0.05:
        verdict = "Improvement"
        recommendation = "Promote changes for further validation"
    elif pf_delta is not None and pf_delta < -0.05:
        verdict = "Regression"
        if n_changes > 1:
            recommendation = "Reject bundled changes — run isolated experiments"
        else:
            recommendation = "Reject this change"
    else:
        verdict = "Neutral"
        recommendation = "Inconclusive — gather more evidence or refine scope"

    confidence = "High" if (pf_delta is not None and abs(pf_delta) >= 0.1) or n_changes == 1 else "Medium"
    if n_changes >= 4 and verdict == "Regression":
        confidence = "High"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasons": reasons,
        "recommendation": recommendation,
    }


def analyze_root_cause(
    comparison: dict[str, Any],
    *,
    template: dict[str, Any],
    verdict: dict[str, Any],
) -> dict[str, Any]:
    causes: list[str] = []
    routing = template.get("routing") or {}
    strategy_changes = routing.get("strategy_changes") or []
    feature_changes = routing.get("feature_changes") or []
    trade_delta_pct = _num(comparison.get("trade_count_delta_pct"))

    for ch in strategy_changes:
        filters = ch.get("filters") or {}
        if filters.get("min_premium") is not None and trade_delta_pct is not None and trade_delta_pct < -10:
            causes.append(
                f"Premium filter (>{filters.get('min_premium')}) likely reduced opportunities "
                f"({trade_delta_pct:.1f}% fewer trades)"
            )
        if filters.get("min_confidence") is not None:
            causes.append("Confidence threshold may exclude entries (requires confidence on prediction rows)")
        if filters.get("stop_pct") is not None:
            causes.append(f"Stop loss at {filters.get('stop_pct')}% changes exit profile")
        if filters.get("max_abs_theta") or filters.get("skip_range"):
            causes.append(f"{ch.get('text')} is analysis-only — not enforced in simulator yet")

    if feature_changes and verdict.get("verdict") != "Improvement":
        causes.append("Model retrain did not offset strategy-level regression in this bundle")

    if not causes:
        causes.append("No dominant root cause identified — consider isolated single-change experiments")

    return {
        "most_likely": causes[0] if causes else "Unknown",
        "factors": causes,
    }


def compute_information_gain(
    comparison: dict[str, Any],
    *,
    template: dict[str, Any],
    verdict: dict[str, Any],
) -> dict[str, Any]:
    n_changes = len(template.get("accepted_changes") or [])
    pf_delta_pct = abs(_num(comparison.get("pf_delta_pct")) or 0)
    trade_delta_pct = abs(_num(comparison.get("trade_count_delta_pct")) or 0)
    verdict_label = verdict.get("verdict") or "Neutral"

    if verdict_label == "Regression" and n_changes >= 3:
        label, score = "Very High", 92
        note = "Bundled regression isolates a bad hypothesis — high learning value"
    elif verdict_label == "Regression" and trade_delta_pct >= 15:
        label, score = "Very High", 88
        note = "Large trade-count shift explains performance change"
    elif verdict_label == "Regression":
        label, score = "High", 75
        note = "Negative result still reduces future search space"
    elif verdict_label == "Improvement" and pf_delta_pct >= 5:
        label, score = "Medium", 60
        note = "Performance gain confirmed"
    elif verdict_label == "Improvement":
        label, score = "Low", 45
        note = "Small improvement — limited new information"
    else:
        label, score = "Low", 35
        note = "Neutral outcome — narrow the hypothesis next"

    return {
        "label": label,
        "score": score,
        "note": note,
        "change_count": n_changes,
    }


def suggest_next_experiments(
    template: dict[str, Any],
    *,
    comparison: dict[str, Any],
    verdict: dict[str, Any],
    information_gain: dict[str, Any],
) -> list[dict[str, Any]]:
    changes = template.get("accepted_changes") or []
    if len(changes) <= 1:
        return []

    suggestions: list[dict[str, Any]] = []
    routing = template.get("routing") or {}
    had_train = bool(routing.get("feature_changes") or routing.get("model_changes"))
    had_strategy = bool(routing.get("strategy_changes"))

    if had_train and had_strategy and verdict.get("verdict") == "Regression":
        suggestions.append({
            "title": "Retrain Only",
            "goal": "Isolate model/feature changes without strategy filter bundle",
            "selection": [c for c in changes if str(c.get("target")) in ("feature_registry", "master_dataset", "model_builder")],
            "expected_information_gain": "Very High",
            "stars": 5,
            "reason": "Previous bundle mixed model + strategy; isolate retrain effect",
        })
        suggestions.append({
            "title": "Strategy Filters Only",
            "goal": "Isolate strategy filters on baseline model predictions",
            "selection": [c for c in changes if str(c.get("target")) == "strategy_registry"],
            "expected_information_gain": "Very High",
            "stars": 5,
            "reason": "Test whether strategy filters alone caused regression",
        })

    for ch in changes:
        target = str(ch.get("target") or "strategy_registry")
        if target == "strategy_registry":
            gain = "High" if verdict.get("verdict") == "Regression" else "Medium"
            stars = 4 if gain == "High" else 3
        else:
            gain = "Very High" if had_strategy else "High"
            stars = 5 if gain == "Very High" else 4
        suggestions.append({
            "title": str(ch.get("text") or "Single change")[:64],
            "goal": f"Isolate: {ch.get('text')}",
            "selection": [ch],
            "expected_information_gain": gain,
            "stars": stars,
            "reason": "Single-change follow-up from bundled experiment",
            "target": target,
        })

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for s in sorted(suggestions, key=lambda x: (-int(x.get("stars") or 0), x.get("title") or "")):
        key = str(s.get("title") or "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique[:6]


def build_job_closure(
    *,
    collected: dict[str, Any],
    comparison: dict[str, Any],
    verdict: dict[str, Any],
    root_cause: dict[str, Any],
    information_gain: dict[str, Any],
    next_experiments: list[dict[str, Any]],
    knowledge: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "completed",
        "verdict": verdict.get("verdict"),
        "verdict_confidence": verdict.get("confidence"),
        "recommendation": verdict.get("recommendation"),
        "information_gain": information_gain,
        "root_cause": root_cause,
        "knowledge_updated": bool(knowledge.get("findings_updated")),
        "findings_updated": knowledge.get("findings_updated") or 0,
        "research_report_saved": bool(collected.get("research_report_id")),
        "next_experiments_count": len(next_experiments),
        "checklist": {
            "job_success": True,
            "outputs_collected": True,
            "baseline_compared": True,
            "verdict_generated": True,
            "research_report": bool(collected.get("research_report_id")),
            "knowledge_extracted": bool(knowledge.get("findings_updated")),
            "next_experiments_generated": len(next_experiments) > 0,
        },
    }


def run_post_job_automation(
    data_dir: str,
    *,
    template: dict[str, Any],
    job: dict[str, Any],
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Full automatic post-job pipeline — called when job execution finishes."""
    from .finding_extraction import extract_findings_from_job

    if job.get("status") == "failed":
        return {
            "ok": False,
            "error": job.get("error") or "job failed",
            "closure": {"status": "failed", "checklist": {"job_success": False}},
        }

    report_doc = ctx.get("research_report") or {}
    exec_sum = report_doc.get("executive_summary") or {}
    ctx = dict(ctx)
    ctx.setdefault("metrics", {})

    collected = collect_job_outputs(template=template, job=job, ctx=ctx)
    collected["after_grade"] = exec_sum.get("overall_grade")

    comparison = compare_with_baseline(collected)
    verdict = generate_experiment_verdict(comparison, collected=collected, template=template)
    comparison["trading_impact"] = build_trading_impact(
        comparison,
        collected=collected,
        template=template,
        verdict=verdict,
        outputs=job.get("outputs") or {},
    )
    root_cause = analyze_root_cause(comparison, template=template, verdict=verdict)
    information_gain = compute_information_gain(comparison, template=template, verdict=verdict)
    next_experiments = suggest_next_experiments(
        template,
        comparison=comparison,
        verdict=verdict,
        information_gain=information_gain,
    )

    knowledge = extract_findings_from_job(
        data_dir,
        template=template,
        job={
            **job,
            "outputs": {
                **(job.get("outputs") or {}),
                **collected,
            },
        },
        comparison={
            **comparison,
            "baseline_grade": comparison.get("baseline_grade"),
            "after_grade": collected.get("after_grade"),
        },
        trade_count=collected.get("strategy", {}).get("trade_count"),
        campaign_id=str(template.get("campaign_id") or job.get("campaign_id") or "") or None,
        program_id=str(template.get("program_id") or job.get("program_id") or "") or None,
    )

    closure = build_job_closure(
        collected=collected,
        comparison=comparison,
        verdict=verdict,
        root_cause=root_cause,
        information_gain=information_gain,
        next_experiments=next_experiments,
        knowledge=knowledge,
    )

    return {
        "ok": True,
        "collected": collected,
        "comparison": comparison,
        "verdict": verdict,
        "root_cause": root_cause,
        "information_gain": information_gain,
        "next_experiments": next_experiments,
        "knowledge": knowledge,
        "closure": closure,
        "decision": default_job_decision(
            verdict,
            template=template,
            outputs=job.get("outputs") or {},
        ),
        "follow_up": recommend_follow_up_experiment(
            data_dir,
            template=template,
            job={
                **job,
                "comparison": comparison,
                "results": {
                    "verdict": verdict,
                    "information_gain": information_gain,
                    "next_experiments": next_experiments,
                    "root_cause": root_cause,
                },
            },
        ),
    }
