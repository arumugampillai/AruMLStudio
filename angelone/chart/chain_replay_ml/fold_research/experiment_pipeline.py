"""Phase A/B — Proposal → Template → Job orchestration."""

from __future__ import annotations

from typing import Any

from .experiment_pipeline_store import JOB_STEPS, STEP_EXPLANATIONS, ExperimentPipelineStore, _utc_now
from .experiment_recommendations import TARGET_LABELS
from .experiment_score import compute_experiment_score
from .experiment_strategy_apply import clone_strategy_for_template


def _route_changes(accepted: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "strategy_changes": [],
        "feature_changes": [],
        "model_changes": [],
        "dataset_changes": [],
        "optimization_changes": [],
    }
    for item in accepted:
        target = str(item.get("target") or "strategy_registry")
        row = dict(item)
        row["target_label"] = item.get("target_label") or TARGET_LABELS.get(target, target)
        if target == "strategy_registry":
            buckets["strategy_changes"].append(row)
        elif target in ("feature_registry", "master_dataset"):
            buckets["feature_changes"].append(row)
        elif target == "dataset_migration":
            buckets["dataset_changes"].append(row)
        elif target == "hyperparameter_optimization":
            buckets["optimization_changes"].append(row)
        elif target == "model_builder":
            buckets["model_changes"].append(row)
        else:
            buckets["strategy_changes"].append(row)
    return buckets


def _baseline_from_report(report: dict[str, Any]) -> dict[str, Any]:
    exec_sum = report.get("executive_summary") or {}
    baseline = report.get("baseline_metrics") or {}
    return {
        "prediction_run_id": report.get("prediction_run_id"),
        "strategy_run_id": report.get("strategy_run_id"),
        "model_id": exec_sum.get("model_id"),
        "strategy_label": exec_sum.get("strategy"),
        "grade": exec_sum.get("overall_grade"),
        "profit_factor": baseline.get("profit_factor"),
        "win_rate_pct": baseline.get("win_rate_pct"),
        "trade_count": exec_sum.get("trade_count") or baseline.get("trade_count"),
    }


def _analyze_template_routing(routing: dict[str, Any]) -> dict[str, Any]:
    has_strategy = bool(routing.get("strategy_changes"))
    phase_c_items = [
        *(routing.get("feature_changes") or []),
        *(routing.get("model_changes") or []),
        *(routing.get("optimization_changes") or []),
    ]
    phase_d_items = routing.get("dataset_changes") or []
    needs_training = bool(phase_c_items)
    needs_dataset_migration = bool(phase_d_items)
    strategy_only = has_strategy and not needs_training and not needs_dataset_migration
    can_run_phase_b = has_strategy
    can_run_phase_c = needs_training
    return {
        "has_strategy": has_strategy,
        "needs_training": needs_training,
        "needs_dataset_migration": needs_dataset_migration,
        "strategy_only": strategy_only,
        "can_run_phase_b": can_run_phase_b,
        "can_run_phase_c": can_run_phase_c,
        "phase_c_items": phase_c_items,
        "phase_d_items": phase_d_items,
        # Backward-compatible keys
        "needs_train": needs_training or needs_dataset_migration,
        "deferred_items": phase_d_items,
        "deferred_by_target": {"dataset_changes": phase_d_items},
    }


def _baseline_strategy_version_id(data_dir: str, template: dict[str, Any]) -> str:
    from chain_replay_ml.strategy_simulator.store import StrategyRunStore

    strategy_run_id = str(template.get("strategy_run_id") or "")
    with StrategyRunStore(data_dir) as run_store:
        baseline_run = run_store.get_run(strategy_run_id)
    if not baseline_run:
        raise ValueError(f"baseline strategy run not found: {strategy_run_id}")
    version_id = str(baseline_run.get("strategy_version_id") or "")
    if not version_id:
        raise ValueError("baseline strategy run missing strategy_version_id")
    return version_id


def _resolve_job_phase(route_info: dict[str, Any]) -> str:
    if route_info.get("can_run_phase_c") and route_info.get("can_run_phase_b"):
        return "B+C"
    if route_info.get("can_run_phase_c"):
        return "C"
    return "B"


def _deferred_summary(deferred_items: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for item in deferred_items:
        label = item.get("target_label") or item.get("target") or "unknown"
        text = str(item.get("text") or "").strip()
        lines.append(f"{text} → {label}" if text else str(label))
    return lines


def _pipeline_steps(*, skip_train: bool) -> list[str]:
    steps: list[str] = []
    for step in JOB_STEPS:
        if skip_train and step in ("training", "walk_forward"):
            continue
        steps.append(step)
    return steps


def _append_activity(
    progress: dict[str, Any],
    *,
    step: str,
    message: str,
    level: str = "info",
) -> None:
    log = list(progress.get("activity_log") or [])
    if log and log[-1].get("message") == message and log[-1].get("step") == step:
        return
    log.append({
        "ts": _utc_now(),
        "step": step,
        "message": message,
        "level": level,
    })
    progress["activity_log"] = log[-200:]


def _save_step(
    store: ExperimentPipelineStore,
    job: dict[str, Any],
    *,
    step: str,
    step_index: int,
    steps: list[str],
    message: str,
    **extra: Any,
) -> None:
    job["current_step"] = step
    progress = dict(job.get("progress") or {})
    progress.update({
        "step_index": step_index,
        "steps": steps,
        "message": message,
        "step_explanation": STEP_EXPLANATIONS.get(step, ""),
        "total_steps": len(steps),
        **extra,
    })
    _append_activity(progress, step=step, message=message)
    job["progress"] = progress
    job["updated_at"] = _utc_now()
    store.save_job(job)


def _fail_job(
    store: ExperimentPipelineStore,
    job: dict[str, Any],
    *,
    step: str,
    error: str,
) -> dict[str, Any]:
    job["status"] = "failed"
    job["current_step"] = step
    job["error"] = error
    job["completed_at"] = _utc_now()
    progress = dict(job.get("progress") or {})
    progress["message"] = error
    progress["step_explanation"] = STEP_EXPLANATIONS.get(step, "")
    _append_activity(progress, step=step, message=error, level="error")
    job["progress"] = progress
    job["updated_at"] = _utc_now()
    saved = store.save_job(job)
    return {"ok": False, "error": error, "job": saved}


def _change_to_planner_item(
    change: dict[str, Any],
    *,
    accepted_default: bool = False,
    source: str = "follow_up",
) -> dict[str, Any]:
    text = str(change.get("text") or "").strip()
    key = str(change.get("key") or text.lower()).strip()
    target = str(change.get("target") or "strategy_registry")
    return {
        "key": key,
        "text": text,
        "source": source,
        "target": target,
        "target_label": change.get("target_label") or TARGET_LABELS.get(target, target),
        "feature_hints": change.get("feature_hints") or [],
        "filters": change.get("filters") or {},
        "accepted_default": accepted_default,
        "stars": int(change.get("stars") or 0),
    }


def _change_match_key(change: dict[str, Any]) -> str:
    from .experiment_similarity import _change_signature

    return _change_signature(change)


def create_proposal_from_suggestion(
    data_dir: str,
    template_id: str,
    suggestion: dict[str, Any],
    *,
    source_job_id: str | None = None,
) -> dict[str, Any]:
    """Create a draft proposal from a post-job next-experiment suggestion."""
    selection = suggestion.get("selection") or []
    if not selection:
        return {"ok": False, "error": "suggestion has no selection"}

    with ExperimentPipelineStore(data_dir) as store:
        template = store._load_template(template_id)
        if not template:
            return {"ok": False, "error": "template not found"}

        parent_changes = template.get("accepted_changes") or []
        if not parent_changes:
            return {"ok": False, "error": "template has no changes to isolate"}

        parent_by_sig = {_change_match_key(ch): ch for ch in parent_changes}
        selected_sigs: set[str] = set()
        selected_items: list[dict[str, Any]] = []
        for ch in selection:
            matched = parent_by_sig.get(_change_match_key(ch)) or ch
            sig = _change_match_key(matched)
            selected_sigs.add(sig)
            selected_items.append(_change_to_planner_item(matched, accepted_default=True))

        seen_keys: set[str] = set()
        deduped_selected: list[dict[str, Any]] = []
        for item in selected_items:
            key = str(item.get("key") or "")
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped_selected.append(item)
        selected_items = deduped_selected
        if not selected_items:
            return {"ok": False, "error": "could not match suggestion to template changes"}

        available_items = [
            _change_to_planner_item(ch, accepted_default=_change_match_key(ch) in selected_sigs)
            for ch in parent_changes
        ]

        goal_text = str(suggestion.get("goal") or suggestion.get("title") or "").strip()
        baseline = template.get("baseline") or {}
        report_stub = {
            "ok": True,
            "research_report_id": template.get("research_report_id"),
            "prediction_run_id": template.get("prediction_run_id"),
            "strategy_run_id": template.get("strategy_run_id"),
            "executive_summary": {
                "model_id": template.get("model_id"),
                "strategy": template.get("strategy_label"),
            },
            "baseline_metrics": {
                "profit_factor": baseline.get("profit_factor"),
                "win_rate_pct": baseline.get("win_rate_pct"),
            },
        }
        score = compute_experiment_score(
            data_dir,
            report_stub,
            accepted_items=selected_items,
            goal=goal_text,
        )
        score["follow_up"] = {
            "parent_template_id": template_id,
            "parent_template_number": template.get("template_number"),
            "source_job_id": source_job_id,
            "suggestion_title": suggestion.get("title"),
            "expected_information_gain": suggestion.get("expected_information_gain"),
            "suggestion_stars": suggestion.get("stars"),
            "reason": suggestion.get("reason"),
        }
        if suggestion.get("stars"):
            score["stars"] = max(int(score.get("stars") or 0), int(suggestion.get("stars") or 0))

        tags = list(dict.fromkeys([*(score.get("tags") or []), "follow_up", "isolated"]))
        proposal = {
            "status": "draft",
            "research_report_id": template.get("research_report_id"),
            "prediction_run_id": template.get("prediction_run_id"),
            "strategy_run_id": template.get("strategy_run_id"),
            "model_id": template.get("model_id"),
            "strategy_label": template.get("strategy_label"),
            "goal": goal_text,
            "tags": tags,
            "available_recommendations": available_items,
            "selected_recommendations": selected_items,
            "baseline": baseline,
            "score": score,
        }
        saved = store.save_proposal(proposal)
    return {"ok": True, "proposal": saved}


def create_proposal_from_report(
    data_dir: str,
    report: dict[str, Any],
    *,
    goal: str | None = None,
) -> dict[str, Any]:
    if not report.get("ok"):
        return report
    from .experiment_planner import build_experiment_planner_view

    view = build_experiment_planner_view(report)
    available = view.get("items") or []
    selected = [i for i in available if i.get("accepted_default")]
    goal_text = goal or view.get("suggested_goal") or ""
    exec_sum = report.get("executive_summary") or {}
    score = compute_experiment_score(data_dir, report, accepted_items=selected, goal=goal_text) if selected else {}

    proposal = {
        "status": "draft",
        "research_report_id": report.get("report_id"),
        "prediction_run_id": report.get("prediction_run_id"),
        "strategy_run_id": report.get("strategy_run_id"),
        "model_id": exec_sum.get("model_id"),
        "strategy_label": exec_sum.get("strategy"),
        "goal": goal_text,
        "tags": score.get("tags") or [],
        "available_recommendations": available,
        "selected_recommendations": selected,
        "baseline": _baseline_from_report(report),
        "score": score,
    }
    with ExperimentPipelineStore(data_dir) as store:
        saved = store.save_proposal(proposal)
    return {"ok": True, "proposal": saved}


def update_proposal_selection(
    data_dir: str,
    proposal_id: str,
    *,
    selected_keys: list[str],
    goal: str | None = None,
) -> dict[str, Any]:
    with ExperimentPipelineStore(data_dir) as store:
        proposal = store._load_proposal(proposal_id)
        if not proposal:
            return {"ok": False, "error": "proposal not found"}
        if proposal.get("status") != "draft":
            return {"ok": False, "error": "proposal is not editable"}

        available = proposal.get("available_recommendations") or []
        key_set = {str(k) for k in selected_keys}
        selected = [
            item for item in available
            if str(item.get("key") or item.get("text")) in key_set
        ]
        if not selected:
            return {"ok": False, "error": "select at least one recommendation"}

        goal_text = goal if goal is not None else proposal.get("goal")
        report_stub = {
            "ok": True,
            "prediction_run_id": proposal.get("prediction_run_id"),
            "strategy_run_id": proposal.get("strategy_run_id"),
            "executive_summary": {
                "model_id": proposal.get("model_id"),
                "strategy": proposal.get("strategy_label"),
            },
            "baseline_metrics": {
                "profit_factor": (proposal.get("baseline") or {}).get("profit_factor"),
                "win_rate_pct": (proposal.get("baseline") or {}).get("win_rate_pct"),
            },
        }
        score = compute_experiment_score(data_dir, report_stub, accepted_items=selected, goal=goal_text)
        proposal["selected_recommendations"] = selected
        proposal["goal"] = goal_text
        proposal["tags"] = score.get("tags") or []
        proposal["score"] = score
        saved = store.save_proposal(proposal)
    return {"ok": True, "proposal": saved}


def create_template_from_proposal(data_dir: str, proposal_id: str) -> dict[str, Any]:
    with ExperimentPipelineStore(data_dir) as store:
        proposal = store._load_proposal(proposal_id)
        if not proposal:
            return {"ok": False, "error": "proposal not found"}
        if proposal.get("status") != "draft":
            return {"ok": False, "error": "proposal already converted"}
        selected = proposal.get("selected_recommendations") or []
        if not selected:
            return {"ok": False, "error": "no selected recommendations"}

        accepted = []
        for item in selected:
            accepted.append({
                "text": item.get("text"),
                "target": item.get("target"),
                "target_label": item.get("target_label"),
                "filters": item.get("filters") or {},
                "feature_hints": item.get("feature_hints") or [],
            })

        template = {
            "proposal_id": proposal_id,
            "research_report_id": proposal.get("research_report_id"),
            "prediction_run_id": proposal.get("prediction_run_id"),
            "strategy_run_id": proposal.get("strategy_run_id"),
            "model_id": proposal.get("model_id"),
            "strategy_label": proposal.get("strategy_label"),
            "goal": proposal.get("goal"),
            "tags": proposal.get("tags") or [],
            "accepted_changes": accepted,
            "routing": _route_changes(accepted),
            "baseline": proposal.get("baseline") or {},
            "score": proposal.get("score") or {},
            "campaign_id": proposal.get("campaign_id"),
            "program_id": proposal.get("program_id"),
            "objective_score": proposal.get("objective_score"),
            "status": "ready",
        }
        saved_template = store.save_template(template)
        proposal["status"] = "converted"
        proposal["template_id"] = saved_template.get("template_id")
        store.save_proposal(proposal)
    return {"ok": True, "template": saved_template}


def create_template_job(
    data_dir: str,
    template_id: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with ExperimentPipelineStore(data_dir) as store:
        template = store._load_template(template_id)
        if not template:
            return {"ok": False, "error": "template not found"}

        job = {
            "template_id": template_id,
            "status": "running",
            "current_step": JOB_STEPS[0],
            "progress": {
                "step_index": 0,
                "steps": list(JOB_STEPS),
                "message": "Job queued — starting pipeline…",
                "step_explanation": STEP_EXPLANATIONS.get(JOB_STEPS[0], ""),
                "activity_log": [{
                    "ts": _utc_now(),
                    "step": JOB_STEPS[0],
                    "message": "Job created and queued for execution.",
                    "level": "info",
                }],
            },
            "overrides": overrides or {},
            "started_at": _utc_now(),
        }
        saved = store.save_job(job)
    return {"ok": True, "job": saved}


def run_template_job(
    data_dir: str,
    template_id: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created = create_template_job(data_dir, template_id, overrides=overrides)
    if not created.get("ok"):
        return created
    job = created.get("job") or {}
    return execute_job_pipeline(data_dir, str(job.get("job_id") or ""))


def execute_job_pipeline(data_dir: str, job_id: str) -> dict[str, Any]:
    """Phase B/C — clone → train (optional) → simulate → report → knowledge base."""
    from chain_replay_ml.strategy_simulator import run_strategy_simulation

    from .experiment_model_apply import prepare_model_training_for_template, run_training_for_template
    from .finding_extraction import extract_findings_from_job
    from .research_report import build_research_report
    from .research_report_store import save_research_report

    ctx: dict[str, Any] = {}

    try:
        with ExperimentPipelineStore(data_dir) as store:
            job = store._load_job(job_id)
            if not job:
                return {"ok": False, "error": "job not found"}
            template = store._load_template(str(job.get("template_id") or ""))
            if not template:
                return _fail_job(store, job, step="preparing", error="template not found")

            routing = template.get("routing") or {}
            route_info = _analyze_template_routing(routing)
            has_strategy = route_info["has_strategy"]
            needs_training = route_info["needs_training"]
            needs_dataset_migration = route_info["needs_dataset_migration"]
            strategy_only = route_info["strategy_only"]
            phase_d_items = route_info["phase_d_items"]
            job_phase = _resolve_job_phase(route_info)
            skip_train = not needs_training
            steps = _pipeline_steps(skip_train=skip_train)

            if not has_strategy and not needs_training:
                if needs_dataset_migration:
                    return _fail_job(
                        store,
                        job,
                        step="preparing",
                        error="Dataset migration only — supported in Phase D.",
                    )
                return _fail_job(store, job, step="preparing", error="Template has no runnable changes.")

            if needs_dataset_migration and not needs_training and not has_strategy:
                return _fail_job(
                    store,
                    job,
                    step="preparing",
                    error="Dataset migration only — supported in Phase D.",
                )

            partial_run = bool(phase_d_items)
            ctx["partial_run"] = partial_run
            ctx["deferred_items"] = phase_d_items
            ctx["job_phase"] = job_phase

            baseline_prediction_run_id = str(template.get("prediction_run_id") or "")
            prediction_run_id = baseline_prediction_run_id
            if not baseline_prediction_run_id and needs_training:
                return _fail_job(store, job, step="preparing", error="template missing baseline prediction_run_id")
            if has_strategy and not template.get("strategy_run_id"):
                return _fail_job(store, job, step="preparing", error="template missing baseline strategy_run_id")
            if needs_training and not template.get("model_id"):
                return _fail_job(store, job, step="preparing", error="template missing baseline model_id")

            baseline = template.get("baseline") or {}
            job_overrides = dict(job.get("overrides") or {})

            for idx, step in enumerate(steps):
                if step == "preparing":
                    prep_msg = "Validating template and baseline run…"
                    if job_phase == "B+C":
                        prep_msg = (
                            f"Phase B+C — training {len(route_info.get('phase_c_items') or [])} model/feature change(s)"
                            f" and {len(routing.get('strategy_changes') or [])} strategy change(s)."
                        )
                    elif job_phase == "C":
                        prep_msg = f"Phase C — training {len(route_info.get('phase_c_items') or [])} change(s)."
                    elif partial_run:
                        prep_msg = (
                            f"Partial run — {len(phase_d_items)} dataset migration item(s) deferred to Phase D."
                        )
                    _save_step(
                        store,
                        job,
                        step=step,
                        step_index=idx,
                        steps=steps,
                        message=prep_msg,
                        strategy_only=strategy_only,
                        partial_run=partial_run,
                        job_phase=job_phase,
                    )

                elif step == "cloning":
                    clone_msgs: list[str] = []
                    if needs_training:
                        _save_step(
                            store,
                            job,
                            step=step,
                            step_index=idx,
                            steps=steps,
                            message="Cloning model training config and applying feature changes…",
                        )
                        model_prep = prepare_model_training_for_template(
                            data_dir,
                            template,
                            overrides=job_overrides.get("training_config"),
                        )
                        ctx["model_prep"] = model_prep
                        ctx["training_config"] = model_prep["training_config"]
                        clone_msgs.extend(model_prep.get("notes") or [])
                        model_name = model_prep.get("model_name") or template.get("model_id") or "model"
                        _save_step(
                            store,
                            job,
                            step=step,
                            step_index=idx,
                            steps=steps,
                            message=(
                                f"Model config cloned from {model_name} · "
                                f"mode={model_prep.get('mode') or 'training'} · "
                                f"{len(model_prep.get('merged_features') or [])} feature hint(s)"
                            ),
                        )

                    if has_strategy:
                        if not needs_training:
                            _save_step(
                                store,
                                job,
                                step=step,
                                step_index=idx,
                                steps=steps,
                                message="Cloning strategy version with experiment filters…",
                            )
                        clone_info = clone_strategy_for_template(data_dir, template)
                        ctx["clone"] = clone_info
                        ctx["strategy_version_id"] = clone_info["strategy_version_id"]
                        clone_msgs.extend(clone_info.get("notes") or [])
                        filters = clone_info.get("applied_filters") or {}
                        filter_bits = ", ".join(f"{k}={v}" for k, v in filters.items()) or "no filter overrides"
                        _save_step(
                            store,
                            job,
                            step=step,
                            step_index=idx,
                            steps=steps,
                            message=(
                                f"Strategy cloned → {clone_info.get('version_label') or clone_info.get('strategy_version_id')} "
                                f"({filter_bits})"
                            ),
                        )

                    ctx["clone_notes"] = clone_msgs
                    if clone_msgs:
                        for note in clone_msgs[:4]:
                            progress = dict(job.get("progress") or {})
                            _append_activity(progress, step=step, message=str(note))
                            job["progress"] = progress
                            store.save_job(job)

                elif step == "training":
                    _save_step(
                        store,
                        job,
                        step=step,
                        step_index=idx,
                        steps=steps,
                        message="Training model and recording walk-forward prediction run…",
                    )
                    training_config = ctx.get("training_config")
                    if not training_config:
                        return _fail_job(store, job, step=step, error="training config missing")

                    def _train_progress(event: dict[str, Any]) -> None:
                        stage = str(event.get("stage") or event.get("step") or "training")
                        msg = str(event.get("message") or event.get("status") or stage)
                        _save_step(
                            store,
                            job,
                            step=step,
                            step_index=idx,
                            steps=steps,
                            message=f"Training: {msg}",
                            training_event=event,
                        )

                    train_out = run_training_for_template(
                        data_dir,
                        template,
                        training_config,
                        on_progress=_train_progress,
                    )
                    if not train_out.get("ok"):
                        err = train_out.get("error") or "training failed"
                        if train_out.get("blocked"):
                            val = train_out.get("validation") or {}
                            gates = val.get("gates") or val.get("blocking_gates") or val
                            err = f"{err} — {gates}"
                        return _fail_job(store, job, step=step, error=str(err))

                    ctx["training_result"] = train_out.get("training_result") or {}
                    ctx["model_name"] = train_out.get("model_name")
                    ctx["prediction_run_id"] = train_out.get("prediction_run_id")
                    prediction_run_id = str(ctx["prediction_run_id"] or baseline_prediction_run_id)

                elif step == "walk_forward":
                    _save_step(
                        store,
                        job,
                        step=step,
                        step_index=idx,
                        steps=steps,
                        message="Walk-forward validation completed during training.",
                    )

                elif step == "simulation":
                    _save_step(
                        store,
                        job,
                        step=step,
                        step_index=idx,
                        steps=steps,
                        message="Running strategy simulation on prediction run…",
                    )
                    sim_pred_id = str(ctx.get("prediction_run_id") or baseline_prediction_run_id)
                    if not sim_pred_id:
                        return _fail_job(store, job, step=step, error="no prediction run available for simulation")
                    strategy_version_id = ctx.get("strategy_version_id")
                    if not strategy_version_id and has_strategy:
                        return _fail_job(store, job, step=step, error="strategy version missing for simulation")
                    if not strategy_version_id:
                        strategy_version_id = _baseline_strategy_version_id(data_dir, template)

                    sim = run_strategy_simulation(
                        data_dir,
                        prediction_run_id=sim_pred_id,
                        strategy_version_id=str(strategy_version_id),
                    )
                    run = sim.get("run") or {}
                    ctx["strategy_run_id"] = run.get("strategy_run_id")
                    metrics = (run.get("metrics") or {}) if isinstance(run.get("metrics"), dict) else {}
                    ctx["metrics"] = metrics
                    prediction_run_id = sim_pred_id
                    pf = metrics.get("profit_factor")
                    trades = metrics.get("trade_count") or metrics.get("total_trades")
                    _save_step(
                        store,
                        job,
                        step=step,
                        step_index=idx,
                        steps=steps,
                        message=(
                            f"Simulation complete · PF={pf if pf is not None else '—'} · "
                            f"trades={trades if trades is not None else '—'} · "
                            f"run={str(run.get('strategy_run_id') or '')[:12]}…"
                        ),
                    )

                elif step == "research_report":
                    _save_step(
                        store,
                        job,
                        step=step,
                        step_index=idx,
                        steps=steps,
                        message="Generating research report for experiment run…",
                    )
                    report_pred_id = str(ctx.get("prediction_run_id") or baseline_prediction_run_id)
                    report = build_research_report(
                        data_dir,
                        report_pred_id,
                        strategy_run_id=str(ctx.get("strategy_run_id") or ""),
                    )
                    if not report.get("ok"):
                        return _fail_job(
                            store,
                            job,
                            step=step,
                            error=report.get("error") or "research report generation failed",
                        )
                    saved_report = save_research_report(data_dir, report)
                    ctx["research_report"] = saved_report
                    ctx["research_report_id"] = saved_report.get("report_id")
                    grade = (report.get("executive_summary") or {}).get("overall_grade") or "—"
                    _save_step(
                        store,
                        job,
                        step=step,
                        step_index=idx,
                        steps=steps,
                        message=f"Research report saved · grade={grade} · id={str(saved_report.get('report_id') or '')[:12]}…",
                    )

                elif step == "knowledge_base":
                    from .experiment_post_job import run_post_job_automation

                    _save_step(
                        store,
                        job,
                        step=step,
                        step_index=idx,
                        steps=steps,
                        message="Auto verdict, knowledge extraction, next experiments…",
                    )
                    post = run_post_job_automation(data_dir, template=template, job=job, ctx=ctx)
                    if not post.get("ok"):
                        return _fail_job(store, job, step=step, error=post.get("error") or "post-job automation failed")
                    ctx["comparison"] = post.get("comparison") or {}
                    ctx["verdict"] = post.get("verdict") or {}
                    ctx["root_cause"] = post.get("root_cause") or {}
                    ctx["information_gain"] = post.get("information_gain") or {}
                    ctx["next_experiments"] = post.get("next_experiments") or []
                    ctx["knowledge"] = post.get("knowledge") or {}
                    ctx["closure"] = post.get("closure") or {}
                    ctx["collected"] = post.get("collected") or {}
                    ctx["decision"] = post.get("decision") or {}
                    ctx["follow_up"] = post.get("follow_up")
                    verdict_label = (post.get("verdict") or {}).get("verdict") or "—"
                    gain_label = (post.get("information_gain") or {}).get("label") or "—"
                    next_count = len(post.get("next_experiments") or [])
                    _save_step(
                        store,
                        job,
                        step=step,
                        step_index=idx,
                        steps=steps,
                        message=(
                            f"Post-job closure · Verdict={verdict_label} · "
                            f"Information Gain={gain_label} · {next_count} next experiment(s)"
                        ),
                    )

                elif step == "complete":
                    _save_step(
                        store,
                        job,
                        step=step,
                        step_index=idx,
                        steps=steps,
                        message="Experiment job complete.",
                    )
                    comparison = ctx.get("comparison") or {}
                    verdict = ctx.get("verdict") or {}
                    closure = ctx.get("closure") or {}
                    clone_info = ctx.get("clone") or {}
                    model_prep = ctx.get("model_prep") or {}
                    notes = list(ctx.get("clone_notes") or [])
                    if partial_run:
                        notes.append(
                            f"Partial run: {len(phase_d_items)} dataset migration item(s) deferred to Phase D."
                        )
                        for line in _deferred_summary(phase_d_items):
                            notes.append(f"Deferred: {line}")
                    job["status"] = "complete"
                    job["current_step"] = "complete"
                    job["completed_at"] = _utc_now()
                    job["comparison"] = comparison
                    job["results"] = {
                        "metrics": ctx.get("metrics") or {},
                        "knowledge": ctx.get("knowledge") or {},
                        "training": ctx.get("training_result") or {},
                        "verdict": verdict,
                        "root_cause": ctx.get("root_cause") or {},
                        "information_gain": ctx.get("information_gain") or {},
                        "next_experiments": ctx.get("next_experiments") or [],
                        "closure": closure,
                        "collected": ctx.get("collected") or {},
                        "decision": ctx.get("decision") or {},
                        "follow_up": ctx.get("follow_up"),
                    }
                    job["outputs"] = {
                        "phase": job_phase,
                        "partial_run": partial_run,
                        "template_id": template.get("template_id"),
                        "template_number": template.get("template_number"),
                        "baseline_prediction_run_id": baseline_prediction_run_id,
                        "prediction_run_id": ctx.get("prediction_run_id") or baseline_prediction_run_id,
                        "model_name": ctx.get("model_name"),
                        "training_mode": model_prep.get("mode"),
                        "merged_features": model_prep.get("merged_features") or [],
                        "strategy_version_id": ctx.get("strategy_version_id"),
                        "strategy_version_label": clone_info.get("version_label"),
                        "strategy_run_id": ctx.get("strategy_run_id"),
                        "research_report_id": ctx.get("research_report_id"),
                        "applied_filters": clone_info.get("applied_filters") or {},
                        "config_overrides": clone_info.get("config_overrides") or {},
                        "deferred_changes": phase_d_items if partial_run else [],
                        "notes": notes,
                    }
                    if job_phase == "B":
                        job["progress"]["message"] = (
                            f"Complete — Verdict: {verdict.get('verdict') or '—'} · "
                            f"Information Gain: {((ctx.get('information_gain') or {}).get('label')) or '—'}"
                        )
                    elif job_phase == "C":
                        job["progress"]["message"] = (
                            f"Complete — Verdict: {verdict.get('verdict') or '—'} · "
                            f"Next experiments: {len(ctx.get('next_experiments') or [])} suggested"
                        )
                    else:
                        job["progress"]["message"] = (
                            f"Complete — Verdict: {verdict.get('verdict') or '—'} · "
                            f"{verdict.get('recommendation') or ''}"
                        )
                    if partial_run:
                        job["progress"]["message"] += " Dataset migration still deferred to Phase D."
                    job["updated_at"] = _utc_now()
                    saved = store.save_job(job)
                    result = {"ok": True, "job": saved}
                    campaign_id = str(template.get("campaign_id") or job.get("campaign_id") or "")
                    if campaign_id:
                        from .campaign_scheduler import on_campaign_job_complete

                        sched = on_campaign_job_complete(data_dir, campaign_id, result)
                        result["scheduler"] = sched
                    return result

            return _fail_job(store, job, step="complete", error="pipeline ended without completion")

    except Exception as exc:
        with ExperimentPipelineStore(data_dir) as store:
            job = store._load_job(job_id) or {"job_id": job_id}
            return _fail_job(store, job, step=str(job.get("current_step") or "preparing"), error=str(exc))


def _metric_delta(before: Any, after: Any) -> float | None:
    try:
        if before is None or after is None:
            return None
        return round(float(after) - float(before), 4)
    except (TypeError, ValueError):
        return None


def reprocess_experiment_job_closure(data_dir: str, job_id: str) -> dict[str, Any]:
    """Backfill verdict / next experiments for a completed job (no re-execution)."""
    from .experiment_post_job import run_post_job_automation

    with ExperimentPipelineStore(data_dir) as store:
        job = store._load_job(job_id)
        if not job:
            return {"ok": False, "error": "job not found"}
        if job.get("status") != "complete":
            return {"ok": False, "error": "job is not complete"}
        template = store._load_template(str(job.get("template_id") or ""))
        if not template:
            return {"ok": False, "error": "template not found"}

        results = job.get("results") or {}
        comparison = job.get("comparison") or {}
        existing_decision = results.get("decision")
        outputs = job.get("outputs") or {}
        ctx = {
            "metrics": results.get("metrics") or {},
            "training_result": results.get("training") or {},
            "research_report_id": outputs.get("research_report_id"),
            "prediction_run_id": outputs.get("prediction_run_id"),
            "strategy_run_id": outputs.get("strategy_run_id"),
            "model_name": outputs.get("model_name"),
            "strategy_version_id": outputs.get("strategy_version_id"),
            "job_phase": outputs.get("phase"),
            "clone": {
                "version_label": outputs.get("strategy_version_label"),
                "notes": outputs.get("notes") or [],
            },
            "model_prep": {"mode": outputs.get("training_mode")},
        }
        post = run_post_job_automation(data_dir, template=template, job=job, ctx=ctx)
        if not post.get("ok"):
            return post
        job["comparison"] = post.get("comparison") or comparison
        job["results"] = {
            **results,
            "verdict": post.get("verdict") or {},
            "root_cause": post.get("root_cause") or {},
            "information_gain": post.get("information_gain") or {},
            "next_experiments": post.get("next_experiments") or [],
            "closure": post.get("closure") or {},
            "collected": post.get("collected") or {},
            "knowledge": post.get("knowledge") or results.get("knowledge") or {},
            "decision": existing_decision or post.get("decision") or {},
            "follow_up": post.get("follow_up") or results.get("follow_up"),
        }
        saved = store.save_job(job)
    return {"ok": True, "job": saved}


def update_experiment_job_decision(
    data_dir: str,
    job_id: str,
    decision: dict[str, Any],
) -> dict[str, Any]:
    with ExperimentPipelineStore(data_dir) as store:
        job = store._load_job(job_id)
        if not job:
            return {"ok": False, "error": "job not found"}
        if job.get("status") != "complete":
            return {"ok": False, "error": "decision is only available for completed jobs"}
        results = dict(job.get("results") or {})
        current = dict(results.get("decision") or {})
        for key in (
            "promote_strategy",
            "promote_model",
            "archive_as_evidence",
            "repeat_modified_hypothesis",
        ):
            if key in decision:
                current[key] = bool(decision[key])
        if "notes" in decision:
            current["notes"] = str(decision.get("notes") or "")
        current["updated_at"] = _utc_now()
        results["decision"] = current
        job["results"] = results
        saved = store.save_job(job)
    return {"ok": True, "job": saved}


def create_follow_up_template_from_job(data_dir: str, job_id: str) -> dict[str, Any]:
    """One-click follow-up — proposal + frozen template from job learnings."""
    from .experiment_post_job import recommend_follow_up_experiment

    with ExperimentPipelineStore(data_dir) as store:
        job = store._load_job(job_id)
        if not job:
            return {"ok": False, "error": "job not found"}
        if job.get("status") != "complete":
            return {"ok": False, "error": "job is not complete"}
        template = store._load_template(str(job.get("template_id") or ""))
        if not template:
            return {"ok": False, "error": "template not found"}

    suggestion = (job.get("results") or {}).get("follow_up")
    if not suggestion or not suggestion.get("selection"):
        suggestion = recommend_follow_up_experiment(data_dir, template=template, job=job)
    if not suggestion or not suggestion.get("selection"):
        return {"ok": False, "error": "No follow-up experiment recommendation available"}

    out = create_proposal_from_suggestion(
        data_dir,
        str(template.get("template_id") or ""),
        suggestion,
        source_job_id=job_id,
    )
    if not out.get("ok"):
        return out
    proposal = out.get("proposal") or {}
    pid = str(proposal.get("proposal_id") or "")
    return create_template_from_proposal(data_dir, pid)


def list_experiment_proposals(data_dir: str, *, status: str | None = "draft", limit: int = 50) -> list[dict[str, Any]]:
    with ExperimentPipelineStore(data_dir) as store:
        return store.list_proposals(status=status, limit=limit)


def list_experiment_templates(data_dir: str, *, limit: int = 50) -> list[dict[str, Any]]:
    with ExperimentPipelineStore(data_dir) as store:
        templates = store.list_templates(limit=limit)
        for t in templates:
            t["job_stats"] = store.count_jobs_for_template(str(t.get("template_id") or ""))
        return templates


def list_experiment_jobs(data_dir: str, *, template_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    with ExperimentPipelineStore(data_dir) as store:
        return store.list_jobs(template_id=template_id, limit=limit)


def get_experiment_job(data_dir: str, job_id: str) -> dict[str, Any] | None:
    with ExperimentPipelineStore(data_dir) as store:
        return store._load_job(job_id)


def get_experiment_template(data_dir: str, template_id: str) -> dict[str, Any] | None:
    with ExperimentPipelineStore(data_dir) as store:
        t = store._load_template(template_id)
        if t:
            t["job_stats"] = store.count_jobs_for_template(template_id)
        return t


def get_experiment_proposal(data_dir: str, proposal_id: str) -> dict[str, Any] | None:
    with ExperimentPipelineStore(data_dir) as store:
        return store._load_proposal(proposal_id)
