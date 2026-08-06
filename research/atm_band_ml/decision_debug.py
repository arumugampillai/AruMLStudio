"""Decision pipeline debug snapshot (unified live scoring path)."""
from __future__ import annotations

from typing import Any, Sequence


def _scored_row_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    return {
        "symbol": str(getattr(row, "symbol", "") or ""),
        "token": str(getattr(row, "token", "") or ""),
        "strike": getattr(row, "strike", None),
        "option_type": str(getattr(row, "option_type", "") or ""),
        "score": getattr(row, "score", None),
        "scorable": bool(getattr(row, "scorable", False)),
        "delta_band": getattr(row, "delta_band", None),
        "ltp": getattr(row, "ltp", None),
        "p_hit": getattr(row, "P_hit", None),
        "pred_max_return": getattr(row, "pred_max_return", None),
        "pred_min_return": getattr(row, "pred_min_return", None),
        "reason": str(getattr(row, "reason", "") or ""),
    }


def _compare_threshold(score: float | None, threshold: float) -> dict[str, Any]:
    if score is None:
        return {"pass": False, "detail": "score is None"}
    try:
        val = float(score)
    except (TypeError, ValueError):
        return {"pass": False, "detail": "score is non-numeric"}
    ok = val >= float(threshold)
    return {
        "pass": ok,
        "detail": f"{val:g} {'>=' if ok else '<'} {float(threshold):g}",
    }


def build_decision_debug_record(
    *,
    signal_ts: float,
    model_name: str,
    target: str,
    configured_threshold: float,
    skip_reason: str,
    suppressed: bool,
    decision_top: Any | None,
    best_top: Any | None,
    scored: Sequence[Any],
) -> dict[str, Any]:
    """
    Debug record for the unified live scoring path:

    ``_score_band_snapshot_live`` → enriched features + fill_missing → pick_top_scored
    """
    probe_row = decision_top or best_top
    probe_score = getattr(probe_row, "score", None) if probe_row is not None else None
    threshold_cmp = _compare_threshold(probe_score, configured_threshold)

    if target.startswith("future_ltp"):
        score_units = "% predicted change in option LTP over horizon (pred vs current option ltp)"
    else:
        score_units = f"registry target: {target}"

    analysis: list[str] = []
    if decision_top is None and best_top is None:
        analysis.append("No scorable rows in live scoring path.")
    elif decision_top is None and best_top is not None:
        analysis.append(
            f"Best score {getattr(best_top, 'score', '?')} below threshold {configured_threshold:g}."
        )
    elif decision_top is not None:
        analysis.append(
            f"Decision pick: {getattr(decision_top, 'symbol', '?')} "
            f"score {getattr(decision_top, 'score', '?')} (threshold {configured_threshold:g})."
        )
    if probe_row is not None and getattr(probe_row, "reason", ""):
        analysis.append(f"Row note: {probe_row.reason}")

    passed = decision_top is not None and not suppressed

    return {
        "signal_ts": float(signal_ts),
        "model_name": str(model_name or ""),
        "target": str(target or ""),
        "configured_threshold": float(configured_threshold),
        "score_units": score_units,
        "skip_reason": str(skip_reason or ""),
        "suppressed": bool(suppressed),
        "decision": "SKIP" if suppressed else "ENTER",
        "kill_code": "score_below_threshold" if "score >=" in str(skip_reason).lower() else "",
        "scoring_path": {
            "name": "_score_band_snapshot_live (enriched + fill_missing=0.0)",
            "decision_top": _scored_row_dict(decision_top),
            "best_top": _scored_row_dict(best_top),
            "threshold_compare": threshold_cmp,
            "row_count": len(scored or ()),
            "rows": [_scored_row_dict(s) for s in (scored or [])[:12]],
        },
        "probe": {
            "symbol": _scored_row_dict(probe_row).get("symbol"),
            "raw_model_output": None,
            "calculated_score": probe_score,
            "p_hit": _scored_row_dict(probe_row).get("p_hit"),
            "threshold": float(configured_threshold),
            "pass_fail": "PASS" if passed else "FAIL",
            "comparison_detail": threshold_cmp.get("detail") or "",
            "rejection_reason": skip_reason if suppressed else "",
        },
        "pipeline": [
            "1. XGBoost predict → pred_ltp (option LTP after horizon)",
            "2. _registry_score(pred_ltp, current_option_ltp, target) → score %",
            "3. _score_band_snapshot_live (enriched features + fill_missing)",
            "4. pick_top_scored(scored, min_score=threshold) → trade decision + UI",
        ],
        "analysis": analysis,
    }


def empty_decision_debug(*, reason: str = "No evaluation yet") -> dict[str, Any]:
    return {
        "signal_ts": None,
        "model_name": "",
        "target": "",
        "configured_threshold": 3.0,
        "score_units": "",
        "skip_reason": reason,
        "suppressed": False,
        "decision": "—",
        "kill_code": "",
        "scoring_path": {
            "name": "",
            "decision_top": {},
            "best_top": {},
            "threshold_compare": {},
            "row_count": 0,
            "rows": [],
        },
        "probe": {},
        "analysis": [reason],
        "pipeline": [],
    }
