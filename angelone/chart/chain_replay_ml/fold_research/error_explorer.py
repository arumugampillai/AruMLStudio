"""Rank worst predictions for Error Explorer."""

from __future__ import annotations

from typing import Any, Literal

ErrorMode = Literal["absolute", "positive", "negative"]


def rank_prediction_errors(
    rows: list[dict[str, Any]],
    *,
    mode: ErrorMode = "absolute",
    limit: int = 100,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        try:
            err = float(row.get("prediction_error"))
        except (TypeError, ValueError):
            continue
        if mode == "positive":
            score = err
        elif mode == "negative":
            score = -err
        else:
            score = abs(err)
        scored.append((score, row))

    scored.sort(key=lambda t: t[0], reverse=True)
    out: list[dict[str, Any]] = []
    for _score, row in scored[:limit]:
        out.append({
            "prediction_id": row.get("prediction_id"),
            "timestamp": row.get("timestamp"),
            "trading_day": row.get("trading_day"),
            "token": row.get("token"),
            "ltp": row.get("ltp"),
            "predicted_ltp": row.get("predicted_ltp"),
            "actual_ltp": row.get("actual_ltp"),
            "prediction_error": row.get("prediction_error"),
            "abs_error": abs(float(row.get("prediction_error") or 0)),
            "direction_correct": row.get("direction_correct"),
        })
    return out
