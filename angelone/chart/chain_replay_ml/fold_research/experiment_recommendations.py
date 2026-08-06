"""Route research recommendations to the correct subsystem."""

from __future__ import annotations

from typing import Any

TARGET_LABELS: dict[str, str] = {
    "strategy_registry": "Strategy Registry",
    "feature_registry": "Feature Registry / Master Dataset",
    "master_dataset": "Master Dataset",
    "model_builder": "Model Builder",
    "hyperparameter_optimization": "Hyperparameter Optimization",
    "dataset_migration": "Dataset Migration",
}


def categorize_recommendation(text: str, *, premium_threshold: float = 25.0) -> dict[str, Any]:
    raw = str(text or "").strip()
    low = raw.lower()
    target = "strategy_registry"
    feature_hints: list[str] = []
    filters: dict[str, Any] = {}

    if any(k in low for k in ("optuna", "hyperparameter", "hpo", "trial")):
        target = "hyperparameter_optimization"
    elif "migration" in low or "dataset migration" in low:
        target = "dataset_migration"
    elif any(k in low for k in ("retrain", "re-train", "more expiry", "expiry days", "july dataset", "more data")):
        if any(k in low for k in ("theta", "iv ", "regime feature", "add feature")):
            target = "feature_registry"
            if "theta" in low:
                feature_hints.append("theta")
            if "iv" in low:
                feature_hints.append("iv")
        else:
            target = "model_builder"
    elif any(k in low for k in ("add theta", "theta feature", "add iv", "iv feature", "iv expansion")):
        target = "feature_registry"
        if "theta" in low:
            feature_hints.append("theta")
        if "iv" in low:
            feature_hints.append("iv")
    elif "premium" in low and ("below" in low or "<" in low or "avoid" in low):
        target = "strategy_registry"
        filters["min_premium"] = premium_threshold
    elif "confidence" in low:
        target = "strategy_registry"
        filters["min_confidence"] = 70.0
    elif "stop" in low and ("7" in low or "%" in low):
        target = "strategy_registry"
        filters["stop_pct"] = 7.0
    elif "theta" in low and "filter" in low:
        target = "strategy_registry"
        filters["max_abs_theta"] = 0.45
    elif "range" in low:
        target = "strategy_registry"
        filters["skip_range"] = True

    return {
        "text": raw,
        "target": target,
        "target_label": TARGET_LABELS.get(target, target),
        "feature_hints": feature_hints,
        "filters": filters,
    }


def build_planner_items_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge ranked recommendations and action-plan items into planner rows."""
    premium_threshold = float(report.get("premium_threshold") or 25.0)
    seen: set[str] = set()
    items: list[dict[str, Any]] = []

    def _add(text: str, *, source: str, stars: int = 0, observed: int = 0, pf_delta: float = 0.0) -> None:
        norm = str(text or "").strip()
        if not norm:
            return
        key = norm.lower()
        if key in seen:
            return
        seen.add(key)
        cat = categorize_recommendation(norm, premium_threshold=premium_threshold)
        items.append({
            "key": key,
            "text": norm,
            "source": source,
            "target": cat["target"],
            "target_label": cat["target_label"],
            "feature_hints": cat["feature_hints"],
            "filters": cat["filters"],
            "stars": stars,
            "observed_trades": observed,
            "expected_pf_delta": pf_delta,
            "accepted_default": True,
        })

    for rec in report.get("recommendations") or []:
        _add(
            str(rec.get("text") or ""),
            source="recommendation",
            stars=int(rec.get("stars") or 0),
            observed=int(rec.get("observed_trades") or 0),
            pf_delta=float(rec.get("expected_pf_delta") or 0),
        )

    for text in (report.get("action_plan") or {}).get("next_experiment") or []:
        _add(str(text), source="action_plan")

    for sc in (report.get("opportunity_analysis") or {}).get("scenarios") or []:
        if sc.get("key") == "stop_7":
            _add("Use 7% stop loss", source="opportunity")

    return items


def infer_experiment_goal(report: dict[str, Any]) -> str:
    root_items = (report.get("root_cause_analysis") or {}).get("items") or []
    if root_items:
        top = root_items[0]
        label = str(top.get("label") or "failures")
        return f"Reduce {label.lower()} failures"
    exec_sum = report.get("executive_summary") or {}
    grade = exec_sum.get("overall_grade")
    if grade and grade not in ("A+", "A"):
        return f"Improve overall grade from {grade}"
    return "Improve walk-forward performance"
