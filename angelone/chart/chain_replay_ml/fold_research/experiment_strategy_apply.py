"""Apply experiment strategy changes — map recommendation filters to strategy config."""

from __future__ import annotations

from typing import Any

from chain_replay_ml.strategy_registry.service import clone_strategy_version


def merge_strategy_filters(strategy_changes: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for change in strategy_changes:
        for key, value in (change.get("filters") or {}).items():
            merged[key] = value
    return merged


def filters_to_config_overrides(filters: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Map recommendation filter keys to strategy config sections."""
    overrides: dict[str, Any] = {}
    notes: list[str] = []

    if filters.get("min_premium") is not None:
        overrides.setdefault("entry", {})["premium_min"] = float(filters["min_premium"])

    if filters.get("min_confidence") is not None:
        val = float(filters["min_confidence"])
        if val > 1.0:
            val = val / 100.0
        overrides.setdefault("confidence", {})["min_signal_strength"] = val
        overrides.setdefault("confidence", {})["use_model_confidence"] = True
        notes.append(
            "Confidence filter enabled — requires confidence values on prediction rows "
            "(no effect when confidence is missing)."
        )

    if filters.get("stop_pct") is not None:
        overrides.setdefault("stop", {})["stop_loss_pct"] = float(filters["stop_pct"])

    if filters.get("max_abs_theta") is not None:
        notes.append("max_abs_theta is analysis-only until simulator gating is extended.")

    if filters.get("skip_range"):
        notes.append("skip_range is analysis-only until regime gating is extended.")

    return overrides, notes


def clone_strategy_for_template(
    data_dir: str,
    template: dict[str, Any],
) -> dict[str, Any]:
    """Clone baseline strategy version with merged template strategy filters."""
    from chain_replay_ml.strategy_simulator.store import StrategyRunStore

    strategy_run_id = str(template.get("strategy_run_id") or "")
    if not strategy_run_id:
        raise ValueError("template missing baseline strategy_run_id")

    with StrategyRunStore(data_dir) as store:
        baseline_run = store.get_run(strategy_run_id)
    if not baseline_run:
        raise ValueError(f"baseline strategy run not found: {strategy_run_id}")

    source_version_id = str(baseline_run.get("strategy_version_id") or "")
    if not source_version_id:
        raise ValueError("baseline strategy run missing strategy_version_id")

    routing = template.get("routing") or {}
    strategy_changes = routing.get("strategy_changes") or []
    filters = merge_strategy_filters(strategy_changes)
    config_overrides, notes = filters_to_config_overrides(filters)

    tnum = template.get("template_number")
    goal = str(template.get("goal") or "").strip()
    description = goal or f"Experiment template #{tnum}"

    cloned = clone_strategy_version(
        data_dir,
        source_version_id=source_version_id,
        description=description,
        config_overrides=config_overrides or None,
    )
    version_id = str(cloned.get("version_id") or "")
    if not version_id:
        raise ValueError("strategy clone did not return version_id")

    return {
        "strategy_version_id": version_id,
        "strategy_id": cloned.get("strategy_id"),
        "source_version_id": source_version_id,
        "config_overrides": config_overrides,
        "applied_filters": filters,
        "notes": notes,
        "version_label": cloned.get("version_label"),
    }
