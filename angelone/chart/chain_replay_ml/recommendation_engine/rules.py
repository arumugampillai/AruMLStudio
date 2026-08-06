"""Deterministic recommendation rules (pure; no I/O).

Each rule returns zero or more suggestion dicts with fields:
id, category, title, reason, reason_bullets, evidence, evidence_score,
expected_benefit, priority, affected_features.

Planner v2 (``apply_rules`` with refine=True) splits those matches by feature
family into research experiments with ``hypothesis`` / ``family`` / enriched
evidence. Rule matchers themselves are unchanged — no new rules.
"""
from __future__ import annotations

from typing import Any, Callable

from chain_replay_ml.recommendation_engine.config import CATEGORIES, merge_thresholds

RuleFn = Callable[[list[dict[str, Any]], dict[str, Any]], list[dict[str, Any]]]

# Evidence keys included on structured affected_features (when present on the row).
_FEATURE_EVIDENCE_KEYS = (
    "risk_score",
    "rank_gain",
    "ks_statistic",
    "drift",
    "drift_pct",
    "wasserstein_normalized",
    "null_drift_pp",
    "null_pct",
    "gain",
    "risk",
)

# Impact levels for expected_benefit breakdown.
BENEFIT_LEVELS = (
    "high",
    "medium",
    "low",
    "slightly_improved",
    "unknown",
    "none",
)


def _f(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _round_num(val: Any) -> Any:
    f = _f(val)
    if f is None:
        return val
    if abs(f - round(f)) < 1e-9 and abs(f) < 1e12:
        return int(round(f)) if abs(f) >= 1 or f == 0 else round(f, 4)
    return round(f, 4)


def evidence_score_from_unit(confidence_0_1: float) -> int:
    """Map legacy 0–1 confidence to 0–100 evidence score."""
    conf = max(0.0, min(1.0, float(confidence_0_1)))
    return int(round(conf * 100))


def unit_from_evidence_score(score: Any) -> float | None:
    """Map 0–100 evidence_score (or legacy 0–1 confidence) to 0–1 unit."""
    f = _f(score)
    if f is None:
        return None
    if f > 1.0:
        return max(0.0, min(1.0, f / 100.0))
    return max(0.0, min(1.0, f))


def _priority_from_unit(confidence: float, th: dict[str, Any]) -> str:
    if confidence >= float(th["priority_high_confidence_min"]):
        return "High"
    if confidence >= float(th["priority_medium_confidence_min"]):
        return "Medium"
    return "Low"


def feature_names(affected: Any) -> list[str]:
    """Extract feature name strings from list of names or evidence objects."""
    out: list[str] = []
    if not isinstance(affected, list):
        return out
    for item in affected:
        if isinstance(item, dict):
            name = str(item.get("feature") or "").strip()
            if name:
                out.append(name)
        else:
            name = str(item).strip()
            if name:
                out.append(name)
    return out


def _affected_feature_obj(row: dict[str, Any]) -> dict[str, Any]:
    obj: dict[str, Any] = {"feature": str(row["feature"])}
    for key in _FEATURE_EVIDENCE_KEYS:
        if key not in row or row.get(key) is None:
            continue
        val = row[key]
        if key == "risk":
            obj[key] = str(val)
        else:
            obj[key] = _round_num(val)
    return obj


def _benefit(
    *,
    summary: str,
    model_stability: str = "unknown",
    prediction_accuracy: str = "unknown",
    training_speed: str = "unknown",
) -> dict[str, str]:
    return {
        "model_stability": model_stability,
        "prediction_accuracy": prediction_accuracy,
        "training_speed": training_speed,
        "summary": summary,
    }


def _reason_text(bullets: list[str]) -> str:
    return " · ".join(b for b in bullets if b)


def _is_high_drift(row: dict[str, Any], th: dict[str, Any]) -> bool:
    drift = _f(row.get("drift"))
    risk_score = _f(row.get("risk_score"))
    risk = str(row.get("risk") or "").lower()
    if drift is not None and drift >= float(th["high_drift"]):
        return True
    if risk_score is not None and risk_score >= float(th["high_risk_score"]):
        return True
    if risk == str(th["high_risk_label"]).lower():
        return True
    return False


def _is_high_risk(row: dict[str, Any], th: dict[str, Any]) -> bool:
    risk_score = _f(row.get("risk_score"))
    risk = str(row.get("risk") or "").lower()
    if risk == str(th["high_risk_label"]).lower():
        return True
    if risk_score is not None and risk_score >= float(th["high_risk_score"]):
        return True
    return False


def _is_high_importance(row: dict[str, Any], th: dict[str, Any]) -> bool:
    rank = _f(row.get("rank_gain"))
    if rank is not None and rank <= float(th["high_importance_rank_max"]):
        return True
    return False


def _is_low_importance(row: dict[str, Any], th: dict[str, Any], n_features: int) -> bool:
    rank = _f(row.get("rank_gain"))
    low_rank_min = float(th["low_importance_rank_min"])
    # Soft floor when feature count is small: bottom half.
    effective_min = min(low_rank_min, max(2.0, n_features * 0.5))
    if rank is not None and rank >= effective_min:
        return True
    if rank is None:
        gain = _f(row.get("gain"))
        if gain is not None and gain <= float(th["low_importance_gain_max"]):
            return True
    return False


def _suggestion(
    *,
    sid: str,
    category: str,
    title: str,
    reason_bullets: list[str],
    evidence: dict[str, Any],
    confidence: float,
    expected_benefit: dict[str, str],
    affected_features: list[dict[str, Any]],
    th: dict[str, Any],
) -> dict[str, Any]:
    conf = max(0.0, min(1.0, float(confidence)))
    score = evidence_score_from_unit(conf)
    bullets = [str(b).strip() for b in reason_bullets if str(b).strip()]
    return {
        "id": sid,
        "category": category,
        "title": title,
        "reason": _reason_text(bullets),
        "reason_bullets": bullets,
        "evidence": evidence,
        "evidence_score": score,
        # Legacy alias (0–1); prefer evidence_score in UI.
        "confidence": round(conf, 3),
        "expected_benefit": dict(expected_benefit),
        "priority": _priority_from_unit(conf, th),
        "affected_features": list(affected_features),
    }


# --- Individual rules (unit-tested) -----------------------------------------


def rule_high_drift_low_importance(
    rows: list[dict[str, Any]], th: dict[str, Any]
) -> list[dict[str, Any]]:
    """1. High Drift + Low Importance → Feature Review."""
    n = len(rows)
    hits = [
        r
        for r in rows
        if _is_high_drift(r, th) and _is_low_importance(r, th, n)
    ]
    if not hits:
        return []
    feats = [_affected_feature_obj(r) for r in hits]
    max_drift = max((_f(r.get("drift")) or 0.0) for r in hits)
    conf = min(0.95, 0.55 + 0.08 * min(len(hits), 5) + 0.2 * min(max_drift, 1.0))
    count = len(hits)
    return [
        _suggestion(
            sid="R1_high_drift_low_importance",
            category="Feature Review",
            title=f"Review {count} drifting low-importance features",
            reason_bullets=[
                "Low importance",
                "High drift",
                f"{count} features matched",
                "Candidates for scrutiny or removal",
            ],
            evidence={
                "rule": "high_drift_low_importance",
                "feature_count": count,
                "max_drift": round(max_drift, 4),
                "thresholds": {
                    "high_drift": th["high_drift"],
                    "low_importance_rank_min": th["low_importance_rank_min"],
                },
                "samples": [
                    {
                        "feature": r["feature"],
                        "drift": r.get("drift"),
                        "rank_gain": r.get("rank_gain"),
                        "risk_score": r.get("risk_score"),
                    }
                    for r in hits[:10]
                ],
            },
            confidence=conf,
            expected_benefit=_benefit(
                summary="Reduce noise; clarify whether drift is actionable",
                model_stability="medium",
                prediction_accuracy="unknown",
                training_speed="slightly_improved",
            ),
            affected_features=feats,
            th=th,
        )
    ]


def rule_high_null_drift(
    rows: list[dict[str, Any]], th: dict[str, Any]
) -> list[dict[str, Any]]:
    """2. High Null Drift → Data Collection."""
    thresh = float(th["high_null_drift_pp"])
    hits = []
    for r in rows:
        nd = _f(r.get("null_drift_pp"))
        if nd is None:
            # Fallback: large holdout null % alone
            null_pct = _f(r.get("null_pct"))
            if null_pct is not None and null_pct >= thresh:
                hits.append(r)
            continue
        if abs(nd) >= thresh:
            hits.append(r)
    if not hits:
        return []
    feats = [_affected_feature_obj(r) for r in hits]
    max_nd = max(abs(_f(r.get("null_drift_pp")) or _f(r.get("null_pct")) or 0.0) for r in hits)
    conf = min(0.95, 0.60 + 0.05 * min(len(hits), 5) + 0.002 * min(max_nd, 50.0))
    count = len(hits)
    return [
        _suggestion(
            sid="R2_high_null_drift",
            category="Data Collection",
            title=f"Investigate {count} features with null/coverage drift",
            reason_bullets=[
                f"Null-rate drift ≥ {thresh:g} pp",
                f"{count} features matched",
                "Review data collection / pipeline coverage",
            ],
            evidence={
                "rule": "high_null_drift",
                "feature_count": count,
                "max_null_drift_pp": round(max_nd, 3),
                "threshold_pp": thresh,
                "samples": [
                    {
                        "feature": r["feature"],
                        "null_drift_pp": r.get("null_drift_pp"),
                        "null_pct": r.get("null_pct"),
                    }
                    for r in hits[:10]
                ],
            },
            confidence=conf,
            expected_benefit=_benefit(
                summary="Improve feature coverage; reduce silent missingness",
                model_stability="medium",
                prediction_accuracy="unknown",
                training_speed="none",
            ),
            affected_features=feats,
            th=th,
        )
    ]


def rule_high_importance_high_drift(
    rows: list[dict[str, Any]], th: dict[str, Any]
) -> list[dict[str, Any]]:
    """3. High Importance + High Drift → Model Refresh."""
    hits = [
        r
        for r in rows
        if _is_high_importance(r, th) and _is_high_drift(r, th)
    ]
    if not hits:
        return []
    feats = [_affected_feature_obj(r) for r in hits]
    max_rs = max((_f(r.get("risk_score")) or 0.0) for r in hits)
    conf = min(0.97, 0.70 + 0.04 * min(len(hits), 5) + 0.003 * min(max_rs, 40.0))
    count = len(hits)
    return [
        _suggestion(
            sid="R3_high_importance_high_drift",
            category="Model Refresh",
            title=f"Refresh model — {count} important features drifted",
            reason_bullets=[
                "High importance",
                "High drift/risk",
                f"{count} features matched",
                "Consider refresh on fresher windows",
            ],
            evidence={
                "rule": "high_importance_high_drift",
                "feature_count": count,
                "max_risk_score": round(max_rs, 2),
                "thresholds": {
                    "high_importance_rank_max": th["high_importance_rank_max"],
                    "high_drift": th["high_drift"],
                },
                "samples": [
                    {
                        "feature": r["feature"],
                        "rank_gain": r.get("rank_gain"),
                        "drift": r.get("drift"),
                        "risk_score": r.get("risk_score"),
                    }
                    for r in hits[:10]
                ],
            },
            confidence=conf,
            expected_benefit=_benefit(
                summary="Restore predictive alignment with current feature regimes",
                model_stability="high",
                prediction_accuracy="medium",
                training_speed="none",
            ),
            affected_features=feats,
            th=th,
        )
    ]


def rule_high_rank_gain_high_risk(
    rows: list[dict[str, Any]], th: dict[str, Any]
) -> list[dict[str, Any]]:
    """4. High Rank Gain (top importance) + High Risk → Distribution investigation."""
    hits = [
        r
        for r in rows
        if _is_high_importance(r, th) and _is_high_risk(r, th)
    ]
    if not hits:
        return []
    feats = [_affected_feature_obj(r) for r in hits]
    conf = min(0.95, 0.65 + 0.05 * min(len(hits), 5))
    count = len(hits)
    return [
        _suggestion(
            sid="R4_high_rank_high_risk",
            category="Feature Review",
            title=f"Investigate distributions for {count} high-risk top features",
            reason_bullets=[
                "Top-ranked importance",
                "High composite risk",
                f"{count} features matched",
                "Inspect holdout distributions before trusting ranks",
            ],
            evidence={
                "rule": "high_rank_gain_high_risk",
                "feature_count": count,
                "samples": [
                    {
                        "feature": r["feature"],
                        "rank_gain": r.get("rank_gain"),
                        "risk": r.get("risk"),
                        "risk_score": r.get("risk_score"),
                        "null_pct": r.get("null_pct"),
                        "skew": r.get("skew"),
                    }
                    for r in hits[:10]
                ],
            },
            confidence=conf,
            expected_benefit=_benefit(
                summary="Catch distribution shifts that inflate apparent importance",
                model_stability="medium",
                prediction_accuracy="unknown",
                training_speed="none",
            ),
            affected_features=feats,
            th=th,
        )
    ]


def rule_large_ks_small_mean_drift(
    rows: list[dict[str, Any]], th: dict[str, Any]
) -> list[dict[str, Any]]:
    """5. Large KS + Small Mean Drift → Distribution Shape Review."""
    large_ks = float(th["large_ks"])
    small_mean = float(th["small_mean_drift_pct"])
    hits = []
    for r in rows:
        ks = _f(r.get("ks_statistic"))
        dp = _f(r.get("drift_pct"))
        if ks is None or dp is None:
            continue
        if ks >= large_ks and abs(dp) <= small_mean:
            hits.append(r)
    if not hits:
        return []
    feats = [_affected_feature_obj(r) for r in hits]
    max_ks = max((_f(r.get("ks_statistic")) or 0.0) for r in hits)
    conf = min(0.93, 0.58 + 0.25 * min(max_ks, 1.0))
    count = len(hits)
    return [
        _suggestion(
            sid="R5_large_ks_small_mean_drift",
            category="Feature Review",
            title=f"Review {count} high-KS features",
            reason_bullets=[
                f"KS ≥ {large_ks:g}",
                f"|Mean drift %| ≤ {small_mean:g}",
                "Shape/quantile shift (not location)",
                f"{count} features matched",
            ],
            evidence={
                "rule": "large_ks_small_mean_drift",
                "feature_count": count,
                "max_ks": round(max_ks, 4),
                "thresholds": {"large_ks": large_ks, "small_mean_drift_pct": small_mean},
                "samples": [
                    {
                        "feature": r["feature"],
                        "ks_statistic": r.get("ks_statistic"),
                        "drift_pct": r.get("drift_pct"),
                    }
                    for r in hits[:10]
                ],
            },
            confidence=conf,
            expected_benefit=_benefit(
                summary="Target shape-aware transforms or segment handling",
                model_stability="medium",
                prediction_accuracy="unknown",
                training_speed="none",
            ),
            affected_features=feats,
            th=th,
        )
    ]


def rule_high_wasserstein_low_ks(
    rows: list[dict[str, Any]], th: dict[str, Any]
) -> list[dict[str, Any]]:
    """6. High Wasserstein + Low KS → Scale Shift Review."""
    high_w = float(th["high_wasserstein_normalized"])
    small_ks = float(th["small_ks"])
    hits = []
    for r in rows:
        w = _f(r.get("wasserstein_normalized"))
        ks = _f(r.get("ks_statistic"))
        if w is None or ks is None:
            continue
        if w >= high_w and ks <= small_ks:
            hits.append(r)
    if not hits:
        return []
    feats = [_affected_feature_obj(r) for r in hits]
    max_w = max((_f(r.get("wasserstein_normalized")) or 0.0) for r in hits)
    conf = min(0.92, 0.55 + 0.15 * min(max_w / max(high_w, 1e-9), 3.0))
    count = len(hits)
    return [
        _suggestion(
            sid="R6_high_wasserstein_low_ks",
            category="Feature Review",
            title=f"Review {count} scale-shift features",
            reason_bullets=[
                f"Wasserstein ≥ {high_w:g}",
                f"KS ≤ {small_ks:g}",
                "Likely scale/magnitude shift",
                f"{count} features matched",
            ],
            evidence={
                "rule": "high_wasserstein_low_ks",
                "feature_count": count,
                "max_wasserstein_normalized": round(max_w, 4),
                "thresholds": {
                    "high_wasserstein_normalized": high_w,
                    "small_ks": small_ks,
                },
                "samples": [
                    {
                        "feature": r["feature"],
                        "wasserstein_normalized": r.get("wasserstein_normalized"),
                        "ks_statistic": r.get("ks_statistic"),
                        "drift_pct": r.get("drift_pct"),
                    }
                    for r in hits[:10]
                ],
            },
            confidence=conf,
            expected_benefit=_benefit(
                summary="Consider rescaling / regime-aware normalization",
                model_stability="medium",
                prediction_accuracy="unknown",
                training_speed="none",
            ),
            affected_features=feats,
            th=th,
        )
    ]


def rule_feature_removal_candidates(
    rows: list[dict[str, Any]], th: dict[str, Any]
) -> list[dict[str, Any]]:
    """Extra: very low importance + high drift → Feature Removal (advisory)."""
    n = len(rows)
    hits = []
    # Stricter than Feature Review: bottom quartile only.
    bottom_cut = n * 0.75
    for r in rows:
        if not _is_high_drift(r, th):
            continue
        rank = _f(r.get("rank_gain"))
        if rank is not None and n >= 4 and rank >= bottom_cut:
            hits.append(r)
        elif rank is None and (_f(r.get("gain")) or 0.0) <= float(
            th["low_importance_gain_max"]
        ):
            hits.append(r)
    if not hits:
        return []
    feats = [_affected_feature_obj(r) for r in hits]
    conf = min(0.85, 0.50 + 0.04 * min(len(hits), 6))
    count = len(hits)
    return [
        _suggestion(
            sid="R7_feature_removal_candidates",
            category="Feature Removal",
            title=f"Review {count} bottom-ranked features",
            reason_bullets=[
                "Bottom-ranked / near-zero gain",
                "High drift",
                f"{count} features matched",
                "Advisory ablation candidates (do not auto-drop)",
            ],
            evidence={
                "rule": "feature_removal_candidates",
                "feature_count": count,
                "samples": [
                    {
                        "feature": r["feature"],
                        "rank_gain": r.get("rank_gain"),
                        "gain": r.get("gain"),
                        "drift": r.get("drift"),
                    }
                    for r in hits[:10]
                ],
            },
            confidence=conf,
            expected_benefit=_benefit(
                summary="Simplify feature set if ablation confirms no loss",
                model_stability="medium",
                prediction_accuracy="unknown",
                training_speed="slightly_improved",
            ),
            affected_features=feats,
            th=th,
        )
    ]


def rule_diagnostics_retraining(
    diagnostics_summary: dict[str, Any] | None, th: dict[str, Any]
) -> list[dict[str, Any]]:
    """Extra: Diagnostics primary_cause → Retraining / Threshold Review."""
    if not diagnostics_summary:
        return []
    cause = str(diagnostics_summary.get("primary_cause") or "").strip().lower()
    if not cause or cause in ("stable", "unknown", "insufficient_inputs"):
        return []
    conf_pct = _f(diagnostics_summary.get("confidence_pct"))
    conf = (conf_pct / 100.0) if conf_pct is not None else 0.55
    conf = max(0.45, min(0.90, conf))

    if cause in ("overfitting",):
        return [
            _suggestion(
                sid="R8_retraining_overfitting",
                category="Retraining",
                title="Consider retraining with stronger regularization",
                reason_bullets=[
                    f"Diagnostics primary cause: {cause}",
                    str(diagnostics_summary.get("label") or cause),
                    "Stronger regularization may reduce train/holdout gap",
                ],
                evidence={
                    "rule": "diagnostics_retraining",
                    "primary_cause": cause,
                    "label": diagnostics_summary.get("label"),
                    "confidence_pct": diagnostics_summary.get("confidence_pct"),
                    "likely_reason": diagnostics_summary.get("likely_reason"),
                },
                confidence=conf,
                expected_benefit=_benefit(
                    summary="Lower overfitting / train–holdout gap",
                    model_stability="high",
                    prediction_accuracy="unknown",
                    training_speed="none",
                ),
                affected_features=[],
                th=th,
            )
        ]
    if cause in ("data_drift", "difficult_market"):
        return [
            _suggestion(
                sid="R8_threshold_review_degradation",
                category="Threshold Review",
                title="Review decision thresholds under degradation",
                reason_bullets=[
                    f"Diagnostics primary cause: {cause}",
                    "Holdout degradation detected",
                    "Review premium / gate thresholds before new experiments",
                ],
                evidence={
                    "rule": "diagnostics_threshold_review",
                    "primary_cause": cause,
                    "label": diagnostics_summary.get("label"),
                    "mae_pct_change": diagnostics_summary.get("mae_pct_change"),
                    "feature_drift_pct": diagnostics_summary.get("feature_drift_pct"),
                },
                confidence=conf,
                expected_benefit=_benefit(
                    summary="Align operating thresholds with observed holdout error",
                    model_stability="medium",
                    prediction_accuracy="unknown",
                    training_speed="none",
                ),
                affected_features=[],
                th=th,
            )
        ]
    return []


def rule_feature_addition_hint(
    rows: list[dict[str, Any]],
    diagnostics_summary: dict[str, Any] | None,
    th: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extra: many high-risk features + data_drift → Feature Addition advisory."""
    if not diagnostics_summary:
        return []
    cause = str(diagnostics_summary.get("primary_cause") or "").strip().lower()
    if cause != "data_drift":
        return []
    high_risk = [r for r in rows if _is_high_risk(r, th)]
    if len(high_risk) < 3:
        return []
    conf = min(0.70, 0.45 + 0.03 * min(len(high_risk), 8))
    count = len(high_risk)
    return [
        _suggestion(
            sid="R9_feature_addition_hint",
            category="Feature Addition",
            title=f"Explore regime features ({count} high-risk)",
            reason_bullets=[
                "Data-drift diagnosis",
                f"{count} high-risk features",
                "Consider regime, session, or coverage signals (advisory)",
            ],
            evidence={
                "rule": "feature_addition_hint",
                "high_risk_count": count,
                "primary_cause": cause,
            },
            confidence=conf,
            expected_benefit=_benefit(
                summary="Capture structural shifts not in the current feature set",
                model_stability="unknown",
                prediction_accuracy="unknown",
                training_speed="none",
            ),
            affected_features=[],
            th=th,
        )
    ]


RULE_ORDER: list[tuple[str, RuleFn]] = [
    ("high_drift_low_importance", rule_high_drift_low_importance),
    ("high_null_drift", rule_high_null_drift),
    ("high_importance_high_drift", rule_high_importance_high_drift),
    ("high_rank_gain_high_risk", rule_high_rank_gain_high_risk),
    ("large_ks_small_mean_drift", rule_large_ks_small_mean_drift),
    ("high_wasserstein_low_ks", rule_high_wasserstein_low_ks),
    ("feature_removal_candidates", rule_feature_removal_candidates),
]


def apply_rules(
    rows: list[dict[str, Any]],
    *,
    thresholds: dict[str, Any] | None = None,
    diagnostics_summary: dict[str, Any] | None = None,
    family_by_name: dict[str, str] | None = None,
    refine: bool = True,
) -> list[dict[str, Any]]:
    """Run all rules; optionally refine into family-split research experiments.

    Rule matchers themselves are unchanged. When ``refine`` is True (default),
    matches are split by feature family, ranked, and capped (Planner v2).
    Pass ``refine=False`` to inspect raw one-suggestion-per-rule output.
    """
    th = merge_thresholds(thresholds)
    suggestions: list[dict[str, Any]] = []
    for _name, fn in RULE_ORDER:
        suggestions.extend(fn(rows, th))
    suggestions.extend(rule_diagnostics_retraining(diagnostics_summary, th))
    suggestions.extend(rule_feature_addition_hint(rows, diagnostics_summary, th))

    # Deduplicate by id (keep first).
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for s in suggestions:
        sid = str(s.get("id") or "")
        if sid in seen:
            continue
        seen.add(sid)
        # Validate category is known.
        if s.get("category") not in CATEGORIES:
            s = {**s, "category": "Feature Review"}
        unique.append(s)

    if refine:
        from chain_replay_ml.recommendation_engine.experiments import (
            refine_to_experiments,
        )

        return refine_to_experiments(
            unique,
            thresholds=th,
            family_by_name=family_by_name,
        )

    pri_rank = {"High": 0, "Medium": 1, "Low": 2}

    def sort_key(s: dict[str, Any]) -> tuple:
        score = _f(s.get("evidence_score"))
        if score is None:
            unit = unit_from_evidence_score(s.get("confidence"))
            score = (unit or 0.0) * 100.0
        return (
            pri_rank.get(str(s.get("priority") or ""), 9),
            -float(score),
            str(s.get("id") or ""),
        )

    unique.sort(key=sort_key)
    return unique


def build_summary(suggestions: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    high = sum(1 for s in suggestions if s.get("priority") == "High")
    medium = sum(1 for s in suggestions if s.get("priority") == "Medium")
    low = sum(1 for s in suggestions if s.get("priority") == "Low")

    highest_risk_feature = None
    best_rs = -1.0
    for r in rows:
        rs = _f(r.get("risk_score"))
        if rs is not None and rs > best_rs:
            best_rs = rs
            highest_risk_feature = str(r.get("feature") or "")

    highest_ev = None
    best_score = -1.0
    for s in suggestions:
        score = _f(s.get("evidence_score"))
        if score is None:
            unit = unit_from_evidence_score(s.get("confidence"))
            score = (unit * 100.0) if unit is not None else None
        if score is not None and score > best_score:
            best_score = score
            highest_ev = {
                "id": s.get("id"),
                "title": s.get("title"),
                "evidence_score": int(round(score)),
                "confidence": round(score / 100.0, 3),
                "priority": s.get("priority"),
                "category": s.get("category"),
            }

    return {
        "total_suggestions": len(suggestions),
        "high_priority": high,
        "medium_priority": medium,
        "low_priority": low,
        "highest_risk_feature": highest_risk_feature,
        "highest_risk_score": round(best_rs, 2) if best_rs >= 0 else None,
        "highest_evidence_suggestion": highest_ev,
        # Legacy alias for older UI/loaders.
        "highest_confidence_suggestion": highest_ev,
        "categories_present": sorted({str(s.get("category")) for s in suggestions}),
    }
