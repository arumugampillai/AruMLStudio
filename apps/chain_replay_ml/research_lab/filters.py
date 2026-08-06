"""Filter normalization for research queries."""

from __future__ import annotations

from typing import Any


def normalize_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    src = dict(filters or {})
    return {
        "model_id": str(src["model_id"]).strip() if src.get("model_id") else None,
        "strategy_id": str(src["strategy_id"]).strip() if src.get("strategy_id") else None,
        "prediction_run_id": str(src["prediction_run_id"]).strip() if src.get("prediction_run_id") else None,
        "target": str(src["target"]).strip() if src.get("target") else None,
        "dataset_name": str(src["dataset_name"]).strip() if src.get("dataset_name") else None,
        "fold_number": int(src["fold_number"]) if src.get("fold_number") is not None else None,
        "min_trades": int(src["min_trades"]) if src.get("min_trades") is not None else None,
        "status": str(src.get("status") or "completed").strip(),
    }


def row_matches_filters(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    if filters.get("status") and row.get("status") != filters["status"]:
        return False
    if filters.get("model_id") and row.get("model_id") != filters["model_id"]:
        return False
    if filters.get("strategy_id") and row.get("strategy_id") != filters["strategy_id"]:
        return False
    if filters.get("prediction_run_id") and row.get("prediction_run_id") != filters["prediction_run_id"]:
        return False
    if filters.get("target") and row.get("target") != filters["target"]:
        return False
    if filters.get("dataset_name") and row.get("dataset_name") != filters["dataset_name"]:
        return False
    if filters.get("fold_number") is not None:
        fn = row.get("fold_number")
        if fn is not None and int(fn) != int(filters["fold_number"]):
            return False
    if filters.get("min_trades") is not None:
        if int(row.get("trade_count") or 0) < int(filters["min_trades"]):
            return False
    return True
