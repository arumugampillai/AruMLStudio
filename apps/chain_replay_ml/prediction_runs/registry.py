"""Read API for prediction runs — list, detail, compare."""

from __future__ import annotations

from typing import Any

from .store import PredictionRunStore


def list_runs(data_dir: str, model_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
    with PredictionRunStore(data_dir) as store:
        return store.list_runs_for_model(model_id, limit=limit)


def list_all_runs(data_dir: str, *, limit: int = 100) -> list[dict[str, Any]]:
    with PredictionRunStore(data_dir) as store:
        return store.list_all_runs(limit=limit)


def get_run_detail(data_dir: str, run_id: str) -> dict[str, Any] | None:
    with PredictionRunStore(data_dir) as store:
        run = store.get_run(run_id)
        if not run:
            return None
        folds = store.list_folds(run_id)
        run["folds"] = folds
        run["prediction_count_stored"] = store.count_rows(run_id)
        return run


def get_fold_rows(
    data_dir: str,
    run_id: str,
    fold_id: str,
    *,
    limit: int = 500,
    offset: int = 0,
) -> dict[str, Any]:
    with PredictionRunStore(data_dir) as store:
        run = store.get_run(run_id)
        if not run:
            return {"ok": False, "error": "run not found"}
        folds = {f["fold_id"]: f for f in store.list_folds(run_id)}
        if fold_id not in folds:
            return {"ok": False, "error": "fold not found"}
        total = store.count_rows(run_id, fold_id=fold_id)
        rows = store.list_rows(run_id, fold_id=fold_id, limit=limit, offset=offset)
        return {
            "ok": True,
            "run": run,
            "fold": folds[fold_id],
            "rows": rows,
            "total": total,
            "limit": limit,
            "offset": offset,
        }


def compare_runs(data_dir: str, run_a: str, run_b: str) -> dict[str, Any]:
    with PredictionRunStore(data_dir) as store:
        a = store.get_run(run_a)
        b = store.get_run(run_b)
        if not a or not b:
            return {"ok": False, "error": "one or both runs not found"}
        folds_a = store.list_folds(run_a)
        folds_b = store.list_folds(run_b)
        by_num_b = {int(f["fold_number"]): f for f in folds_b}
        comparisons: list[dict[str, Any]] = []
        for fa in folds_a:
            fn = int(fa["fold_number"])
            fb = by_num_b.get(fn)
            if not fb:
                continue
            comparisons.append({
                "fold_number": fn,
                "run_a": {
                    "run_id": run_a,
                    "mae": fa.get("mae"),
                    "rmse": fa.get("rmse"),
                    "directional_accuracy_pct": fa.get("directional_accuracy_pct"),
                    "prediction_count": fa.get("prediction_count"),
                },
                "run_b": {
                    "run_id": run_b,
                    "mae": fb.get("mae"),
                    "rmse": fb.get("rmse"),
                    "directional_accuracy_pct": fb.get("directional_accuracy_pct"),
                    "prediction_count": fb.get("prediction_count"),
                },
            })
        return {
            "ok": True,
            "run_a": a,
            "run_b": b,
            "fold_comparisons": comparisons,
        }
