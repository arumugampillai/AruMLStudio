"""Build Diagnostics Studio summary + narrative from joined signals."""

from __future__ import annotations

from typing import Any

from chain_replay_ml.training.holdout_performance import (
    build_prediction_error_change_row,
    diagnose_degradation,
    extract_saved_prediction_metrics,
)


def build_summary_and_narrative(
    *,
    doc: dict[str, Any],
    drift_rows: list[dict[str, Any]],
    drift_meta: dict[str, Any],
    comparison: list[dict[str, Any]],
    joins: dict[str, bool],
) -> tuple[dict[str, Any], list[str]]:
    saved = extract_saved_prediction_metrics(doc)
    production_wf = saved.get("production_wf") or {}
    holdout_test = saved.get("holdout_test") or {}
    metrics_present = bool(production_wf) or bool(holdout_test)
    joins = dict(joins)
    joins["metrics"] = metrics_present

    change = build_prediction_error_change_row(production_wf, holdout_test)

    drift_scores = drift_meta.get("drift_scores") if isinstance(drift_meta.get("drift_scores"), dict) else {}
    similarity_pct = drift_meta.get("similarity_pct")
    if similarity_pct is None and drift_scores:
        # soft fallback: 100 - feature drift
        try:
            similarity_pct = max(0.0, 100.0 - float(drift_scores.get("feature") or 0))
        except (TypeError, ValueError):
            similarity_pct = None

    top_risk = [
        {
            "feature": r.get("feature"),
            "risk": r.get("risk"),
            "risk_score": r.get("risk_score"),
            "rank_gain": r.get("rank_gain"),
        }
        for r in comparison
        if r.get("diagnostic_flag") == "high_risk" or r.get("risk") == "high"
    ][:5]
    if not top_risk:
        top_risk = [
            {
                "feature": r.get("feature"),
                "risk": r.get("risk"),
                "risk_score": r.get("risk_score"),
                "rank_gain": r.get("rank_gain"),
            }
            for r in comparison[:5]
        ]

    enough_for_diagnosis = joins.get("drift") or (
        production_wf.get("mae") is not None and holdout_test.get("mae") is not None
    )

    if enough_for_diagnosis:
        diagnosis = diagnose_degradation(
            wf_validation_mae=_f(production_wf.get("mae")),
            holdout_mae=_f(holdout_test.get("mae")),
            comparison_rows=[],
            wf_target_std=None,
            holdout_target_std=None,
            drift_scores=drift_scores if isinstance(drift_scores, dict) else {},
            feature_ranking=drift_rows or comparison,
            premium_pct_change=_f(change.get("premium_mae_pct_change")),
            similarity_pct=_f(similarity_pct),
        )
    else:
        diagnosis = {
            "primary_cause": "insufficient_inputs",
            "label": "Insufficient inputs",
            "confidence_pct": 0,
            "evidence": [],
            "likely_reason": "Run Feature Drift Studio and ensure package metrics exist.",
            "error_ratio": None,
        }

    summary = {
        "primary_cause": diagnosis.get("primary_cause"),
        "label": diagnosis.get("label"),
        "confidence_pct": diagnosis.get("confidence_pct"),
        "likely_reason": diagnosis.get("likely_reason"),
        "error_ratio": diagnosis.get("error_ratio"),
        "similarity_pct": similarity_pct,
        "drift_scores": drift_scores,
        "feature_drift_pct": drift_meta.get("feature_drift_pct")
        if drift_meta.get("feature_drift_pct") is not None
        else (drift_scores.get("feature") if isinstance(drift_scores, dict) else None),
        "target_drift_pct": drift_meta.get("target_drift_pct")
        if drift_meta.get("target_drift_pct") is not None
        else (drift_scores.get("target") if isinstance(drift_scores, dict) else None),
        "mae_pct_change": change.get("mae_pct_change"),
        "rmse_pct_change": change.get("rmse_pct_change"),
        "premium_mae_pct_change": change.get("premium_mae_pct_change"),
        "premium_rmse_pct_change": change.get("premium_rmse_pct_change"),
        "direction_pts_change": change.get("direction_pts_change"),
        "production_wf": production_wf,
        "holdout_test": holdout_test,
        "top_risk_features": top_risk,
        "joins": joins,
        "evidence": diagnosis.get("evidence") or [],
    }

    narrative = _build_narrative(summary, joins)
    return summary, narrative


def _f(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _build_narrative(summary: dict[str, Any], joins: dict[str, bool]) -> list[str]:
    bullets: list[str] = []
    cause = summary.get("label") or summary.get("primary_cause") or "Unknown"
    conf = summary.get("confidence_pct")
    if summary.get("primary_cause") == "insufficient_inputs":
        bullets.append(
            "Primary diagnosis unavailable — need Drift studio and/or saved WF+holdout MAE."
        )
    else:
        conf_s = f" ({conf}% confidence)" if conf is not None else ""
        bullets.append(f"Primary cause: {cause}{conf_s}.")
        reason = summary.get("likely_reason")
        if reason:
            bullets.append(str(reason))

    sim = summary.get("similarity_pct")
    fd = summary.get("feature_drift_pct")
    if sim is not None or fd is not None:
        parts = []
        if sim is not None:
            parts.append(f"similarity {float(sim):.1f}%")
        if fd is not None:
            parts.append(f"feature drift {float(fd):.1f}%")
        bullets.append("Holdout vs WF: " + ", ".join(parts) + ".")

    mae = summary.get("mae_pct_change")
    if mae is not None:
        bullets.append(f"MAE change (holdout vs production WF): {float(mae):+.1f}%.")
    prmse = summary.get("premium_rmse_pct_change")
    if prmse is not None:
        bullets.append(f"Premium RMSE change: {float(prmse):+.1f}%.")
    direction = summary.get("direction_pts_change")
    if direction is not None:
        bullets.append(f"Directional accuracy Δ: {float(direction):+.1f} pts.")

    top = summary.get("top_risk_features") or []
    if top:
        names = ", ".join(
            f"{r.get('feature')}[{r.get('risk') or '?'}]" for r in top[:3] if r.get("feature")
        )
        if names:
            bullets.append(f"Top risk features: {names}.")

    missing = [k for k, v in joins.items() if not v and k != "metrics"]
    if missing:
        bullets.append(
            "Missing studio joins: "
            + ", ".join(missing)
            + " — run those studios for a fuller picture."
        )
    if not joins.get("metrics"):
        bullets.append("Package metrics incomplete — metric deltas may be empty.")

    bullets.append(
        "Deep premium / top-1 error investigation remains in Registry → Holdout Performance."
    )
    return bullets[:8]
