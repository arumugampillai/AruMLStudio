"""Build model × strategy research matrix from stored runs."""

from __future__ import annotations

from typing import Any

from chain_replay_ml.prediction_runs.store import PredictionRunStore
from chain_replay_ml.strategy_registry.store import StrategyRegistryStore
from chain_replay_ml.strategy_simulator.store import StrategyRunStore

from .filters import normalize_filters, row_matches_filters
from .scoring import composite_research_score, stability_score


def _avg_prediction_fold_metrics(folds: list[dict[str, Any]]) -> dict[str, float | None]:
    if not folds:
        return {"mae": None, "rmse": None, "directional_accuracy_pct": None}
    maes = [float(f["mae"]) for f in folds if f.get("mae") is not None]
    rmses = [float(f["rmse"]) for f in folds if f.get("rmse") is not None]
    dirs = [float(f["directional_accuracy_pct"]) for f in folds if f.get("directional_accuracy_pct") is not None]
    return {
        "mae": round(sum(maes) / len(maes), 4) if maes else None,
        "rmse": round(sum(rmses) / len(rmses), 4) if rmses else None,
        "directional_accuracy_pct": round(sum(dirs) / len(dirs), 2) if dirs else None,
    }


def build_research_matrix(
    data_dir: str,
    *,
    filters: dict[str, Any] | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """Rows = strategy runs enriched with model + strategy + prediction quality."""
    filt = normalize_filters(filters)
    rows: list[dict[str, Any]] = []

    with StrategyRunStore(data_dir) as sr_store, PredictionRunStore(data_dir) as pr_store, StrategyRegistryStore(data_dir) as st_store:
        strategy_runs = sr_store.list_runs(limit=limit)
        pred_cache: dict[str, dict[str, Any] | None] = {}
        strat_cache: dict[str, dict[str, Any] | None] = {}
        version_cache: dict[str, dict[str, Any] | None] = {}

        for sr in strategy_runs:
            metrics = sr.get("metrics") or {}
            pred_id = str(sr.get("prediction_run_id") or "")
            if pred_id not in pred_cache:
                run = pr_store.get_run(pred_id)
                if run:
                    folds = pr_store.list_folds(pred_id)
                    run = dict(run)
                    run["_fold_metrics_avg"] = _avg_prediction_fold_metrics(folds)
                pred_cache[pred_id] = run
            pred_run = pred_cache.get(pred_id)

            sid = str(sr.get("strategy_id") or "")
            if sid not in strat_cache:
                strat_cache[sid] = st_store.get_profile(sid)
            profile = strat_cache.get(sid)

            vid = str(sr.get("strategy_version_id") or "")
            if vid not in version_cache:
                version_cache[vid] = st_store.get_version(vid)
            version = version_cache.get(vid)

            pred_avg = (pred_run or {}).get("_fold_metrics_avg") or {}
            fold_metrics = metrics.get("fold_metrics") or []

            row = {
                "strategy_run_id": sr.get("strategy_run_id"),
                "prediction_run_id": pred_id,
                "strategy_id": sid,
                "strategy_version_id": vid,
                "strategy_name": (profile or {}).get("display_name") or (version or {}).get("display_name"),
                "strategy_version_label": (version or {}).get("version_label"),
                "strategy_config_hash": sr.get("strategy_config_hash"),
                "model_id": sr.get("model_id") or (pred_run or {}).get("model_id"),
                "target": (pred_run or {}).get("target"),
                "dataset_name": (pred_run or {}).get("dataset_name"),
                "scope": sr.get("scope"),
                "fold_number": sr.get("fold_number"),
                "status": sr.get("status"),
                "created_on": sr.get("created_on"),
                "trade_count": metrics.get("trade_count") or sr.get("trade_count"),
                "profit": metrics.get("profit"),
                "win_rate_pct": metrics.get("win_rate_pct"),
                "profit_factor": metrics.get("profit_factor"),
                "max_drawdown": metrics.get("max_drawdown"),
                "avg_return_pct": metrics.get("avg_return_pct"),
                "avg_holding_sec": metrics.get("avg_holding_sec"),
                "prediction_mae": pred_avg.get("mae"),
                "prediction_rmse": pred_avg.get("rmse"),
                "prediction_direction_pct": pred_avg.get("directional_accuracy_pct"),
                "composite_score": composite_research_score(metrics),
                "stability_score": stability_score(fold_metrics),
            }
            if row_matches_filters(row, filt):
                rows.append(row)

    models = sorted({str(r["model_id"]) for r in rows if r.get("model_id")})
    strategies = sorted({str(r["strategy_name"]) for r in rows if r.get("strategy_name")})

    return {
        "ok": True,
        "filters": filt,
        "row_count": len(rows),
        "models": models,
        "strategies": strategies,
        "rows": rows,
    }


def build_model_strategy_grid(matrix_doc: dict[str, Any]) -> dict[str, Any]:
    """Pivot matrix: model rows × strategy columns → profit (or composite)."""
    rows = matrix_doc.get("rows") or []
    models = matrix_doc.get("models") or []
    strategies = matrix_doc.get("strategies") or []
    grid: dict[str, dict[str, Any]] = {}
    for model in models:
        grid[model] = {}
        for strat in strategies:
            grid[model][strat] = None

    for row in rows:
        model = str(row.get("model_id") or "")
        strat = str(row.get("strategy_name") or "")
        if model not in grid or strat not in grid[model]:
            continue
        cell = grid[model][strat]
        score = row.get("composite_score")
        if cell is None or (score is not None and (cell.get("composite_score") or -1e9) < score):
            grid[model][strat] = {
                "strategy_run_id": row.get("strategy_run_id"),
                "profit": row.get("profit"),
                "win_rate_pct": row.get("win_rate_pct"),
                "profit_factor": row.get("profit_factor"),
                "max_drawdown": row.get("max_drawdown"),
                "trade_count": row.get("trade_count"),
                "composite_score": score,
            }

    return {
        "ok": True,
        "models": models,
        "strategies": strategies,
        "grid": grid,
    }
