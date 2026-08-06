"""Fold Research & Replay — Phase 5 service."""

from __future__ import annotations

from typing import Any

from chain_replay_ml.prediction_runs.store import PredictionRunStore
from chain_replay_ml.strategy_simulator.metrics import compute_trade_metrics
from chain_replay_ml.strategy_simulator.store import StrategyRunStore

from .bookmarks_store import list_research_bookmarks
from .fold_quality import compute_fold_quality
from .trade_clusters import discover_trade_clusters
from .chart_series import build_chart_series
from .error_explorer import rank_prediction_errors
from .feature_drift import compute_fold_feature_drift
from .market_summary import summarize_market_context
from .notes_store import list_fold_notes
from .prediction_analysis import analyze_prediction_rows
from .regime_iv import analyze_iv_regimes, load_validation_feature_map
from .replay import build_replay_timeline
from .trade_replay import build_trade_replay


def _list_strategy_runs_for_fold(
    data_dir: str,
    prediction_run_id: str,
    fold_id: str,
) -> list[dict[str, Any]]:
    with StrategyRunStore(data_dir) as store:
        runs = store.list_runs(prediction_run_id=prediction_run_id, limit=100)
        out: list[dict[str, Any]] = []
        for run in runs:
            trade_count = store.count_trades_for_fold(run["strategy_run_id"], fold_id)
            if trade_count > 0 or (
                run.get("scope") == "single_fold" and run.get("fold_id") == fold_id
            ):
                doc = dict(run)
                doc["fold_trade_count"] = trade_count
                out.append(doc)
        return out


def _load_fold_trades(
    data_dir: str,
    strategy_run_id: str,
    fold_id: str,
) -> list[dict[str, Any]]:
    with StrategyRunStore(data_dir) as store:
        all_trades = store.list_trades(strategy_run_id, limit=5000)
    return [t for t in all_trades if str(t.get("fold_id") or "") == fold_id]


def get_fold_research(
    data_dir: str,
    *,
    prediction_run_id: str,
    fold_id: str,
    strategy_run_id: str | None = None,
) -> dict[str, Any]:
    with PredictionRunStore(data_dir) as store:
        pred_run = store.get_run(prediction_run_id)
        if not pred_run:
            return {"ok": False, "error": "prediction run not found"}
        folds = {f["fold_id"]: f for f in store.list_folds(prediction_run_id)}
        fold = folds.get(fold_id)
        if not fold:
            return {"ok": False, "error": "fold not found"}
        rows = store.list_all_rows(prediction_run_id, fold_id=fold_id)

    market = summarize_market_context(rows)
    prediction_quality = analyze_prediction_rows(rows)

    available_runs = _list_strategy_runs_for_fold(data_dir, prediction_run_id, fold_id)
    trades: list[dict[str, Any]] = []
    selected_run = strategy_run_id
    if not selected_run and available_runs:
        selected_run = available_runs[0].get("strategy_run_id")
    if selected_run:
        trades = _load_fold_trades(data_dir, selected_run, fold_id)

    trading_metrics = compute_trade_metrics(trades) if trades else None

    timeline = build_replay_timeline(rows, trades=trades)
    feature_drift = compute_fold_feature_drift(data_dir, run=pred_run, fold=fold)
    error_explorer = {
        "absolute": rank_prediction_errors(rows, mode="absolute", limit=100),
        "positive": rank_prediction_errors(rows, mode="positive", limit=100),
        "negative": rank_prediction_errors(rows, mode="negative", limit=100),
    }
    regime_analysis = analyze_iv_regimes(rows, load_validation_feature_map(data_dir, run=pred_run, fold=fold))
    fold_notes = list_fold_notes(data_dir, prediction_run_id=prediction_run_id, fold_id=fold_id)
    bookmarks = list_research_bookmarks(data_dir, prediction_run_id=prediction_run_id, fold_id=fold_id)
    fold_quality = compute_fold_quality(
        prediction_quality=prediction_quality,
        trading_metrics=trading_metrics,
        regime_analysis=regime_analysis,
        trade_count=len(trades),
    )
    trade_clusters = discover_trade_clusters(trades)

    return {
        "ok": True,
        "prediction_run": pred_run,
        "fold": fold,
        "market_summary": market,
        "prediction_quality": prediction_quality,
        "chart_series": build_chart_series(rows, trades),
        "trading": {
            "strategy_run_id": selected_run,
            "trade_count": len(trades),
            "metrics": trading_metrics,
            "trades": trades[:100],
        } if trades or selected_run else None,
        "strategy_runs_available": available_runs,
        "feature_drift": feature_drift,
        "error_explorer": error_explorer,
        "regime_analysis": regime_analysis,
        "notes": fold_notes,
        "bookmarks": bookmarks,
        "fold_quality": fold_quality,
        "trade_clusters": trade_clusters,
        "replay": {
            "event_count": len(timeline),
            "timeline_preview": timeline[:200],
        },
    }


def get_fold_replay_timeline(
    data_dir: str,
    *,
    prediction_run_id: str,
    fold_id: str,
    strategy_run_id: str | None = None,
    limit: int = 1000,
    offset: int = 0,
) -> dict[str, Any]:
    with PredictionRunStore(data_dir) as store:
        rows = store.list_all_rows(prediction_run_id, fold_id=fold_id)

    trades: list[dict[str, Any]] = []
    if strategy_run_id:
        trades = _load_fold_trades(data_dir, strategy_run_id, fold_id)

    timeline = build_replay_timeline(rows, trades=trades)
    total = len(timeline)
    page = timeline[offset : offset + limit]

    return {
        "ok": True,
        "prediction_run_id": prediction_run_id,
        "fold_id": fold_id,
        "strategy_run_id": strategy_run_id,
        "total": total,
        "limit": limit,
        "offset": offset,
        "events": page,
    }


def list_folds_for_replay(data_dir: str, prediction_run_id: str) -> dict[str, Any]:
    with PredictionRunStore(data_dir) as store:
        run = store.get_run(prediction_run_id)
        if not run:
            return {"ok": False, "error": "prediction run not found"}
        folds = store.list_folds(prediction_run_id)
    return {"ok": True, "prediction_run": run, "folds": folds}


def compare_folds_for_run(
    data_dir: str,
    prediction_run_id: str,
    *,
    strategy_run_id: str | None = None,
) -> dict[str, Any]:
    """Side-by-side fold metrics for comparison table."""
    with PredictionRunStore(data_dir) as store:
        pred_run = store.get_run(prediction_run_id)
        if not pred_run:
            return {"ok": False, "error": "prediction run not found"}
        folds = store.list_folds(prediction_run_id)

    rows_out: list[dict[str, Any]] = []
    for fold in sorted(folds, key=lambda f: int(f.get("fold_number") or 0)):
        fold_id = str(fold.get("fold_id") or "")
        with PredictionRunStore(data_dir) as store:
            pred_rows = store.list_all_rows(prediction_run_id, fold_id=fold_id)
        pq = analyze_prediction_rows(pred_rows)

        trades: list[dict[str, Any]] = []
        selected_run = strategy_run_id
        if selected_run:
            trades = _load_fold_trades(data_dir, selected_run, fold_id)
        elif available := _list_strategy_runs_for_fold(data_dir, prediction_run_id, fold_id):
            selected_run = available[0].get("strategy_run_id")
            if selected_run:
                trades = _load_fold_trades(data_dir, selected_run, fold_id)

        tm = compute_trade_metrics(trades) if trades else None
        rows_out.append({
            "fold_number": fold.get("fold_number"),
            "fold_id": fold_id,
            "validation_rows": fold.get("validation_rows") or pq.get("row_count"),
            "mae": fold.get("mae") if fold.get("mae") is not None else pq.get("mae"),
            "rmse": fold.get("rmse") if fold.get("rmse") is not None else pq.get("rmse"),
            "directional_accuracy_pct": (
                fold.get("directional_accuracy_pct") if fold.get("directional_accuracy_pct") is not None
                else pq.get("directional_accuracy_pct")
            ),
            "bias": pq.get("bias"),
            "trade_count": tm.get("trade_count") if tm else 0,
            "profit": tm.get("profit") if tm else None,
            "profit_factor": tm.get("profit_factor") if tm else None,
            "max_drawdown": tm.get("max_drawdown") if tm else None,
        })

    return {"ok": True, "prediction_run_id": prediction_run_id, "folds": rows_out}


def get_trade_replay(
    data_dir: str,
    *,
    prediction_run_id: str,
    fold_id: str,
    trade_id: str,
) -> dict[str, Any]:
    from chain_replay_ml.strategy_registry.store import StrategyRegistryStore

    with PredictionRunStore(data_dir) as store:
        rows = store.list_all_rows(prediction_run_id, fold_id=fold_id)

    trade: dict[str, Any] | None = None
    with StrategyRunStore(data_dir) as sstore:
        for run in sstore.list_runs(prediction_run_id=prediction_run_id, limit=100):
            for t in sstore.list_trades(run["strategy_run_id"], limit=5000):
                if str(t.get("trade_id") or "") == trade_id:
                    trade = t
                    break
            if trade:
                break

    if not trade:
        return {"ok": False, "error": "trade not found"}

    strategy_run_id = trade.get("strategy_run_id")
    peer_trades: list[dict[str, Any]] = []
    if strategy_run_id:
        peer_trades = _load_fold_trades(data_dir, str(strategy_run_id), fold_id)

    cfg: dict[str, Any] | None = None
    version_id = trade.get("strategy_version_id")
    if version_id:
        with StrategyRegistryStore(data_dir) as rstore:
            ver = rstore.get_version(str(version_id))
            if ver:
                cfg = ver.get("config")

    with PredictionRunStore(data_dir) as store:
        run = store.get_run(prediction_run_id) or {}
        fold = next((f for f in store.list_folds(prediction_run_id) if f["fold_id"] == fold_id), {})

    feature_map = load_validation_feature_map(data_dir, run=run, fold=fold) if run and fold else None
    replay = build_trade_replay(trade, rows, cfg=cfg, feature_rows=feature_map, peer_trades=peer_trades)
    replay["prediction_run_id"] = prediction_run_id
    replay["fold_id"] = fold_id
    return replay


def get_prediction_run_summary(
    data_dir: str,
    prediction_run_id: str,
    *,
    strategy_run_id: str | None = None,
) -> dict[str, Any]:
    from .prediction_run_summary import build_prediction_run_summary

    return build_prediction_run_summary(
        data_dir,
        prediction_run_id,
        strategy_run_id=strategy_run_id,
    )


def get_research_report(
    data_dir: str,
    prediction_run_id: str,
    *,
    strategy_run_id: str | None = None,
) -> dict[str, Any]:
    from .research_report import build_research_report

    return build_research_report(
        data_dir,
        prediction_run_id,
        strategy_run_id=strategy_run_id,
    )


def save_research_report_to_store(data_dir: str, report: dict[str, Any]) -> dict[str, Any]:
    from .research_report_store import save_research_report

    return save_research_report(data_dir, report)


def list_saved_research_reports(
    data_dir: str,
    *,
    prediction_run_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    from .research_report_store import list_research_reports

    return list_research_reports(data_dir, prediction_run_id=prediction_run_id, limit=limit)


def load_saved_research_report(data_dir: str, report_id: str) -> dict[str, Any] | None:
    from .research_report_store import get_research_report

    return get_research_report(data_dir, report_id)


def get_experiment_planner_view(
    data_dir: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    from .experiment_planner import build_experiment_planner_view

    return build_experiment_planner_view(report)


def create_experiment_from_report(
    data_dir: str,
    report: dict[str, Any],
    *,
    accepted_items: list[dict[str, Any]],
    goal: str | None = None,
) -> dict[str, Any]:
    from .experiment_planner import create_experiment_from_report as _create

    return _create(data_dir, report, accepted_items=accepted_items, goal=goal)


def list_experiments(data_dir: str, *, limit: int = 50) -> list[dict[str, Any]]:
    from .experiment_store import list_experiments as _list

    return _list(data_dir, limit=limit)


def get_experiment(data_dir: str, experiment_id: str) -> dict[str, Any] | None:
    from .experiment_store import get_experiment as _get

    return _get(data_dir, experiment_id)


def launch_experiment(data_dir: str, experiment_id: str) -> dict[str, Any]:
    from .experiment_planner import launch_experiment as _launch

    return _launch(data_dir, experiment_id)


def complete_experiment(
    data_dir: str,
    experiment_id: str,
    *,
    results: dict[str, Any],
) -> dict[str, Any]:
    from .experiment_planner import complete_experiment as _complete

    return _complete(data_dir, experiment_id, results=results)


def list_knowledge_findings(
    data_dir: str,
    *,
    status: str | None = None,
    category: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    from .knowledge_store import list_knowledge_findings as _list

    return _list(data_dir, status=status, category=category, limit=limit)


def get_knowledge_finding(data_dir: str, finding_id: str) -> dict[str, Any] | None:
    from .knowledge_store import get_knowledge_finding as _get

    return _get(data_dir, finding_id)


def check_similar_experiments(
    data_dir: str,
    *,
    accepted_items: list[dict[str, Any]],
    goal: str | None = None,
    model_id: str | None = None,
    exclude_experiment_id: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    from .experiment_similarity import check_similar_experiments as _check

    return _check(
        data_dir,
        accepted_items=accepted_items,
        goal=goal,
        model_id=model_id,
        exclude_experiment_id=exclude_experiment_id,
        limit=limit,
    )


def check_experiment_before_create(
    data_dir: str,
    report: dict[str, Any],
    *,
    accepted_items: list[dict[str, Any]],
    goal: str | None = None,
) -> dict[str, Any]:
    from .experiment_similarity import check_experiment_before_create as _check

    return _check(data_dir, report, accepted_items=accepted_items, goal=goal)


def check_experiment_before_launch(data_dir: str, experiment_id: str) -> dict[str, Any]:
    from .experiment_similarity import check_experiment_before_launch as _check

    return _check(data_dir, experiment_id)


def get_known_findings_for_report(data_dir: str, report: dict[str, Any]) -> dict[str, Any]:
    from .knowledge_retrieval import get_known_findings_for_report as _get

    return _get(data_dir, report)


def score_experiment_proposal(
    data_dir: str,
    report: dict[str, Any],
    *,
    accepted_items: list[dict[str, Any]],
    goal: str | None = None,
) -> dict[str, Any]:
    from .knowledge_retrieval import score_experiment_proposal as _score

    return _score(data_dir, report, accepted_items=accepted_items, goal=goal)


def get_feature_knowledge(data_dir: str, feature_names: list[str]) -> list[dict[str, Any]]:
    from .knowledge_retrieval import get_feature_knowledge as _get

    return _get(data_dir, feature_names)


def get_strategy_filter_knowledge(data_dir: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    from .knowledge_retrieval import get_strategy_filter_knowledge as _get

    return _get(data_dir, config)


def create_experiment_proposal_from_report(
    data_dir: str,
    report: dict[str, Any],
    *,
    goal: str | None = None,
) -> dict[str, Any]:
    from .experiment_pipeline import create_proposal_from_report as _create

    return _create(data_dir, report, goal=goal)


def create_experiment_proposal_from_suggestion(
    data_dir: str,
    template_id: str,
    suggestion: dict[str, Any],
    *,
    source_job_id: str | None = None,
) -> dict[str, Any]:
    from .experiment_pipeline import create_proposal_from_suggestion as _create

    return _create(data_dir, template_id, suggestion, source_job_id=source_job_id)


def update_experiment_proposal_selection(
    data_dir: str,
    proposal_id: str,
    *,
    selected_keys: list[str],
    goal: str | None = None,
) -> dict[str, Any]:
    from .experiment_pipeline import update_proposal_selection as _update

    return _update(data_dir, proposal_id, selected_keys=selected_keys, goal=goal)


def create_experiment_template_from_proposal(data_dir: str, proposal_id: str) -> dict[str, Any]:
    from .experiment_pipeline import create_template_from_proposal as _create

    return _create(data_dir, proposal_id)


def run_experiment_template_job(
    data_dir: str,
    template_id: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .experiment_pipeline import run_template_job as _run

    return _run(data_dir, template_id, overrides=overrides)


def list_experiment_proposals(data_dir: str, *, status: str | None = "draft", limit: int = 50) -> list[dict[str, Any]]:
    from .experiment_pipeline import list_experiment_proposals as _list

    return _list(data_dir, status=status, limit=limit)


def list_experiment_templates(data_dir: str, *, limit: int = 50) -> list[dict[str, Any]]:
    from .experiment_pipeline import list_experiment_templates as _list

    return _list(data_dir, limit=limit)


def list_experiment_jobs(
    data_dir: str,
    *,
    template_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    from .experiment_pipeline import list_experiment_jobs as _list

    return _list(data_dir, template_id=template_id, limit=limit)


def get_experiment_proposal(data_dir: str, proposal_id: str) -> dict[str, Any] | None:
    from .experiment_pipeline import get_experiment_proposal as _get

    return _get(data_dir, proposal_id)


def get_experiment_template(data_dir: str, template_id: str) -> dict[str, Any] | None:
    from .experiment_pipeline import get_experiment_template as _get

    return _get(data_dir, template_id)


def get_experiment_job(data_dir: str, job_id: str) -> dict[str, Any] | None:
    from .experiment_pipeline import get_experiment_job as _get

    return _get(data_dir, job_id)


def update_experiment_job_decision(
    data_dir: str,
    job_id: str,
    decision: dict[str, Any],
) -> dict[str, Any]:
    from .experiment_pipeline import update_experiment_job_decision as _update

    return _update(data_dir, job_id, decision)


def create_follow_up_template_from_job(data_dir: str, job_id: str) -> dict[str, Any]:
    from .experiment_pipeline import create_follow_up_template_from_job as _create

    return _create(data_dir, job_id)


def create_research_program(
    data_dir: str,
    *,
    name: str,
    description: str | None = None,
    importance: str = "medium",
    objective: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .research_program import create_research_program as _create

    return _create(
        data_dir,
        name=name,
        description=description,
        importance=importance,
        objective=objective,
        budget=budget,
    )


def get_research_program(data_dir: str, program_id: str) -> dict[str, Any] | None:
    from .research_program import get_research_program as _get

    return _get(data_dir, program_id)


def list_research_programs(data_dir: str, *, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    from .research_program import list_research_programs as _list

    return _list(data_dir, status=status, limit=limit)


def create_research_campaign(
    data_dir: str,
    program_id: str,
    *,
    name: str,
    research_question: str,
    description: str | None = None,
    importance: str | None = None,
    objective: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
    dependencies: list[str] | None = None,
) -> dict[str, Any]:
    from .research_program import create_research_campaign as _create

    return _create(
        data_dir,
        program_id,
        name=name,
        research_question=research_question,
        description=description,
        importance=importance,
        objective=objective,
        budget=budget,
        dependencies=dependencies,
    )


def get_research_campaign(data_dir: str, campaign_id: str) -> dict[str, Any] | None:
    from .research_program import get_research_campaign as _get

    return _get(data_dir, campaign_id)


def list_research_campaigns(
    data_dir: str,
    *,
    program_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    from .research_program import list_research_campaigns as _list

    return _list(data_dir, program_id=program_id, status=status, limit=limit)


def start_research_campaign(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .research_program import start_research_campaign as _start

    return _start(data_dir, campaign_id)


def retire_research_campaign(data_dir: str, campaign_id: str, *, reason: str) -> dict[str, Any]:
    from .research_program import retire_research_campaign as _retire

    return _retire(data_dir, campaign_id, reason=reason)


def get_campaign_config(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .research_program import get_campaign_config as _get

    return _get(data_dir, campaign_id)


def get_campaign_config(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .research_program import get_campaign_config as _get

    return _get(data_dir, campaign_id)


def attach_campaign_baseline(
    data_dir: str,
    campaign_id: str,
    *,
    research_report_id: str,
) -> dict[str, Any]:
    from .campaign_proposal_generator import attach_campaign_baseline as _attach

    return _attach(data_dir, campaign_id, research_report_id=research_report_id)


def seed_campaign_proposals(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .campaign_proposal_generator import seed_proposals_from_report as _seed

    return _seed(data_dir, campaign_id)


def rank_campaign_proposals(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .campaign_proposal_generator import rank_campaign_proposals as _rank

    return _rank(data_dir, campaign_id)


def get_campaign_scheduler_view(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .campaign_scheduler import get_campaign_scheduler_view as _view

    return _view(data_dir, campaign_id)


def run_next_campaign_experiment(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .campaign_scheduler import run_next_campaign_experiment as _run

    return _run(data_dir, campaign_id)


def set_campaign_auto_run(data_dir: str, campaign_id: str, *, enabled: bool) -> dict[str, Any]:
    from .campaign_scheduler import set_campaign_auto_run as _set

    return _set(data_dir, campaign_id, enabled=enabled)


def evaluate_campaign_generalization(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .generalization_score import evaluate_campaign_best_generalization as _eval

    return _eval(data_dir, campaign_id)


def mark_campaign_validated(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .campaign_scheduler import mark_campaign_validated as _mark

    return _mark(data_dir, campaign_id)


def get_research_cycle_view(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .research_cycle import get_cycle_view
    from .research_program import get_research_campaign

    campaign = get_research_campaign(data_dir, campaign_id)
    if not campaign:
        return {"ok": False, "error": "campaign not found"}
    memory = campaign.get("memory") or {}
    return {"ok": True, "cycle": get_cycle_view(memory)}


def get_research_portfolio(data_dir: str, *, limit: int = 50) -> dict[str, Any]:
    from .program_portfolio import get_research_portfolio as _portfolio

    return _portfolio(data_dir, limit=limit)


def get_program_portfolio(data_dir: str, program_id: str) -> dict[str, Any]:
    from .program_portfolio import get_program_portfolio as _portfolio

    return _portfolio(data_dir, program_id)


def get_campaign_dashboard(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .campaign_dashboard import get_campaign_dashboard as _dashboard

    return _dashboard(data_dir, campaign_id)


def build_campaign_report(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .campaign_report import build_campaign_report as _build

    return _build(data_dir, campaign_id)


def get_campaign_outcome(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .campaign_outcome import get_campaign_outcome as _outcome

    return _outcome(data_dir, campaign_id)


def get_campaign_report(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .campaign_report import get_campaign_report as _get

    return _get(data_dir, campaign_id)


def get_program_champion_view(data_dir: str, program_id: str) -> dict[str, Any]:
    from .program_champion import get_program_champion_view as _view

    return _view(data_dir, program_id)


def approve_program_champion(data_dir: str, program_id: str, *, note: str | None = None) -> dict[str, Any]:
    from .program_champion import approve_program_champion as _approve

    return _approve(data_dir, program_id, note=note)


def dismiss_program_champion_candidate(data_dir: str, program_id: str) -> dict[str, Any]:
    from .program_champion import dismiss_program_champion_candidate as _dismiss

    return _dismiss(data_dir, program_id)


def refresh_program_champion_candidate(data_dir: str, program_id: str) -> dict[str, Any]:
    from .program_champion import refresh_program_champion_candidate as _refresh

    return _refresh(data_dir, program_id)


def get_knowledge_pipeline_view(
    data_dir: str,
    *,
    program_id: str | None = None,
    campaign_id: str | None = None,
    limit: int = 30,
) -> dict[str, Any]:
    from .knowledge_pipeline import get_knowledge_pipeline_view as _view

    return _view(data_dir, program_id=program_id, campaign_id=campaign_id, limit=limit)


def promote_finding_to_knowledge(data_dir: str, finding_id: str) -> dict[str, Any]:
    from .knowledge_pipeline import promote_finding_to_knowledge as _promote

    return _promote(data_dir, finding_id)


def get_knowledge_gaps(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .kb_proposal_generator import get_knowledge_gaps_for_campaign as _gaps

    return _gaps(data_dir, campaign_id)


def seed_kb_campaign_proposals(data_dir: str, campaign_id: str, *, limit: int = 4) -> dict[str, Any]:
    from .kb_proposal_generator import seed_kb_driven_proposals as _seed

    return _seed(data_dir, campaign_id, limit=limit)


def compute_objective_score(
    data_dir: str,
    *,
    proposal: dict[str, Any],
    objective: dict[str, Any],
    importance: str = "medium",
    campaign_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .objective_score import compute_objective_score as _score

    return _score(
        data_dir,
        proposal=proposal,
        objective=objective,
        importance=importance,
        campaign_memory=campaign_memory,
    )


def compute_experiment_score(
    data_dir: str,
    report: dict[str, Any],
    *,
    accepted_items: list[dict[str, Any]],
    goal: str | None = None,
) -> dict[str, Any]:
    from .experiment_score import compute_experiment_score as _score

    return _score(data_dir, report, accepted_items=accepted_items, goal=goal)


def start_program_on_model(
    data_dir: str,
    *,
    model_id: str,
    program_id: str,
    research_report_id: str | None = None,
    prediction_run_id: str | None = None,
    strategy_run_id: str | None = None,
    campaign_ids: list[str] | None = None,
) -> dict[str, Any]:
    from .program_runner import start_program_on_model as _start

    return _start(
        data_dir,
        model_id=model_id,
        program_id=program_id,
        research_report_id=research_report_id,
        prediction_run_id=prediction_run_id,
        strategy_run_id=strategy_run_id,
        campaign_ids=campaign_ids,
    )


def resume_program_run(data_dir: str, run_id: str) -> dict[str, Any]:
    from .program_runner import resume_program_run as _resume

    return _resume(data_dir, run_id)


def get_model_research_view(data_dir: str, model_id: str) -> dict[str, Any]:
    from .model_research import get_model_research_view as _view

    return _view(data_dir, model_id)


def build_model_certification(data_dir: str, model_id: str) -> dict[str, Any]:
    from .model_certification import build_model_certification as _cert

    return _cert(data_dir, model_id)


def build_model_research_portfolio_report(data_dir: str, model_id: str) -> dict[str, Any]:
    from .research_portfolio_report import build_model_research_portfolio_report as _report

    return _report(data_dir, model_id)


def build_campaign_manifest(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .campaign_manifest import build_campaign_manifest as _build

    return _build(data_dir, campaign_id)


def evaluate_campaign_should_stop(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .campaign_scheduler import evaluate_campaign_should_stop as _eval

    return _eval(data_dir, campaign_id)
