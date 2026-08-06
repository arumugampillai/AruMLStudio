"""Research-experiment refinement for Experiment Planner v2.

Takes raw rule matches (unchanged rules in ``rules.py``) and:
- splits multi-feature matches by feature family into small experiments
- attaches deterministic hypothesis templates (rule + family)
- enriches evidence (rule id/name, thresholds, matched features, top contributors)
- ranks by aggregate evidence and caps to a configurable limit
"""

from __future__ import annotations

from typing import Any

from chain_replay_ml.recommendation_engine.config import (
    DEFAULT_EXPERIMENT_STATUS,
    EFFORT_EASY_MAX_FEATURES,
    EFFORT_HIGH_MIN_FEATURES,
    MODEL_EXPERIMENT_CATEGORIES,
    PLANNER_VERSION,
    merge_thresholds,
)
from chain_replay_ml.recommendation_engine.families import (
    group_features_by_family,
    resolve_feature_family,
)
from chain_replay_ml.recommendation_engine.rules import (
    _f,
    _priority_from_unit,
    feature_names,
    unit_from_evidence_score,
)

# Title verb / pattern by rule name (evidence["rule"]).
_TITLE_BY_RULE: dict[str, str] = {
    "high_drift_low_importance": "Review {family} Features",
    "high_null_drift": "Investigate {family} Coverage",
    "high_importance_high_drift": "Refresh for {family} Drift",
    "high_rank_gain_high_risk": "Inspect {family} Top Features",
    "large_ks_small_mean_drift": "Review {family} Shape Shift",
    "high_wasserstein_low_ks": "Review {family} Scale Shift",
    "feature_removal_candidates": "Ablate {family} Bottom Features",
    "diagnostics_retraining": "Retrain with Stronger Regularization",
    "diagnostics_threshold_review": "Review Decision Thresholds",
    "feature_addition_hint": "Explore Regime Features",
}

# Deterministic hypothesis templates: rule name → format with {family}.
_HYPOTHESIS_BY_RULE: dict[str, str] = {
    "high_drift_low_importance": (
        "Removing these low-importance drifting {family} features may reduce "
        "unnecessary variance without reducing predictive power."
    ),
    "high_null_drift": (
        "Investigating null/coverage drift in {family} features may restore "
        "reliable inputs and reduce silent missingness."
    ),
    "high_importance_high_drift": (
        "Refreshing after reviewing drifted high-importance {family} features "
        "may restore predictive alignment with current regimes."
    ),
    "high_rank_gain_high_risk": (
        "Inspecting holdout distributions for high-risk top-ranked {family} "
        "features may catch shifts that inflate apparent importance."
    ),
    "large_ks_small_mean_drift": (
        "Reviewing high-KS {family} features may reveal shape/quantile shifts "
        "that mean-drift alone misses."
    ),
    "high_wasserstein_low_ks": (
        "Reviewing scale-shift {family} features may motivate rescaling or "
        "regime-aware normalization."
    ),
    "feature_removal_candidates": (
        "Ablating these bottom-ranked drifting {family} features may simplify "
        "the set if ablation confirms no predictive loss."
    ),
    "diagnostics_retraining": (
        "Stronger regularization on retrain may reduce the train/holdout gap "
        "flagged by diagnostics."
    ),
    "diagnostics_threshold_review": (
        "Reviewing decision thresholds under holdout degradation may align "
        "operating points with observed error."
    ),
    "feature_addition_hint": (
        "Exploring regime, session, or coverage signals may capture structural "
        "shifts not represented in the current feature set."
    ),
}

# Short finding / recommendation label when merging multiple rules per family.
_FINDING_LABEL_BY_RULE: dict[str, str] = {
    "high_drift_low_importance": "Review low importance",
    "high_null_drift": "Investigate coverage / null drift",
    "high_importance_high_drift": "Refresh for high-importance drift",
    "high_rank_gain_high_risk": "Inspect top-ranked high-risk",
    "large_ks_small_mean_drift": "Review shape shift",
    "high_wasserstein_low_ks": "Review scale shift",
    "feature_removal_candidates": "Ablate bottom-ranked",
    "diagnostics_retraining": "Retrain with stronger regularization",
    "diagnostics_threshold_review": "Review decision thresholds",
    "feature_addition_hint": "Explore regime features",
}

# Expected experiment (what to run) — distinct from hypothesis (why it may help).
_EXPECTED_EXPERIMENT_BY_RULE: dict[str, str] = {
    "high_drift_low_importance": (
        "Train without {family} family and compare Holdout MAE, Drift, "
        "Selected Features, Walk-forward."
    ),
    "high_null_drift": (
        "Audit {family} coverage / null rates on holdout vs walk-forward, "
        "then retrain only if coverage is restored."
    ),
    "high_importance_high_drift": (
        "Refresh / retrain after reviewing drifted high-importance {family} "
        "features; compare Holdout MAE, Drift, Selected Features, Walk-forward."
    ),
    "high_rank_gain_high_risk": (
        "Holdout distribution check on top-ranked {family} features, then "
        "optional ablation; compare Holdout MAE and Drift."
    ),
    "large_ks_small_mean_drift": (
        "Ablate or reweight high-KS {family} features and compare Holdout MAE, "
        "KS / Drift, Selected Features, Walk-forward."
    ),
    "high_wasserstein_low_ks": (
        "Rescale or regime-normalize {family} features, retrain, and compare "
        "Holdout MAE, Drift, Selected Features."
    ),
    "feature_removal_candidates": (
        "Ablate bottom-ranked drifting {family} features and compare Holdout "
        "MAE, Drift, Selected Features, Walk-forward."
    ),
    "diagnostics_retraining": (
        "Retrain with stronger regularization and compare Holdout MAE, "
        "train/holdout gap, Walk-forward."
    ),
    "diagnostics_threshold_review": (
        "Sweep decision thresholds on holdout and compare operating-point "
        "error vs current thresholds."
    ),
    "feature_addition_hint": (
        "Add candidate regime/session/coverage features, retrain, and compare "
        "Holdout MAE, Drift, Selected Features, Walk-forward."
    ),
}


def experiment_scope(category: str | None, family: str | None = None) -> str:
    """Return ``model`` or ``feature`` for UI grouping (presentation only)."""
    del family  # reserved; category drives Model vs Feature sections
    cat = str(category or "").strip()
    if cat in MODEL_EXPERIMENT_CATEGORIES:
        return "model"
    return "feature"


def estimate_effort(
    *,
    category: str | None,
    feature_count: int,
    family: str | None = None,
) -> str:
    """Heuristic Easy / Medium / High from category + feature count.

    - Retrain / Model Refresh / Data Collection → High
    - Few features (≤``EFFORT_EASY_MAX_FEATURES``) → Easy
    - Large feature set (≥``EFFORT_HIGH_MIN_FEATURES``) → High
    - Otherwise → Medium
    """
    del family
    cat = str(category or "").strip()
    n = max(0, int(feature_count))
    if cat in ("Retraining", "Model Refresh", "Data Collection"):
        return "High"
    if cat in ("Threshold Review", "Feature Addition"):
        return "Medium"
    # Feature Review / Feature Removal / unknown with feature counts
    if n <= EFFORT_EASY_MAX_FEATURES:
        return "Easy"
    if n >= EFFORT_HIGH_MIN_FEATURES:
        return "High"
    return "Medium"


def finding_label(rule_name: str, fallback_title: str | None = None) -> str:
    """Short recommendation bullet for a rule finding."""
    label = _FINDING_LABEL_BY_RULE.get(rule_name)
    if label:
        return label
    title = str(fallback_title or "").strip()
    if title:
        return title
    return rule_name or "Review finding"


def build_expected_experiment(rule_name: str, family: str | None) -> str:
    tmpl = _EXPECTED_EXPERIMENT_BY_RULE.get(rule_name)
    fam = (family or "model").strip() or "model"
    if tmpl:
        try:
            return tmpl.format(family=fam)
        except (KeyError, ValueError):
            return tmpl
    if family:
        return (
            f"Run a focused {fam} ablation / review and compare Holdout MAE, "
            "Drift, Selected Features, Walk-forward."
        )
    return (
        f"Execute the '{rule_name}' research step and compare Holdout MAE, "
        "Drift, and Walk-forward against the current baseline."
    )


def build_suggested_next_steps(
    *,
    category: str | None,
    family: str | None,
    scope: str,
) -> list[str]:
    """Deterministic advisory checklist (not executable automation)."""
    fam = (family or "").strip()
    cat = str(category or "").strip()
    if scope == "model" or cat in MODEL_EXPERIMENT_CATEGORIES:
        steps = [
            "Open Diagnostics",
            "Review Holdout MAE / train-holdout gap",
            "Note baseline Selected Features and Walk-forward metrics",
            "Decide retrain or threshold change manually (no auto-job)",
        ]
        if cat == "Threshold Review":
            steps = [
                "Open Diagnostics",
                "Review decision thresholds under holdout degradation",
                "Compare operating points vs current thresholds",
                "Record chosen thresholds for a future manual retrain",
            ]
        elif cat == "Feature Addition":
            steps = [
                "Open Importance / Drift for regime gaps",
                "List candidate regime or coverage features",
                "Design a small addition experiment",
                "Compare Holdout MAE after a manual retrain",
            ]
        return steps
    # Feature-family experiments
    filter_hint = f"Sort / filter by family ({fam})" if fam else "Sort by family / filter"
    return [
        "Open Importance",
        filter_hint,
        "Export selected features",
        "Create ablation experiment",
    ]


def format_experiment_id(index: int) -> str:
    """Stable display id EXP-001 … (1-based, zero-padded)."""
    return f"EXP-{max(1, int(index)):03d}"


def _highest_risk_feature(feats: list[Any]) -> tuple[str | None, float | None]:
    best_name: str | None = None
    best_risk: float | None = None
    for item in feats:
        if isinstance(item, dict):
            name = str(item.get("feature") or "").strip()
            risk = _feat_metric(item, "risk_score")
        else:
            name = str(item).strip()
            risk = None
        if not name:
            continue
        if risk is None:
            if best_name is None:
                best_name = name
            continue
        if best_risk is None or risk > best_risk:
            best_risk = risk
            best_name = name
    return best_name, best_risk


def _rule_name(suggestion: dict[str, Any]) -> str:
    evidence = suggestion.get("evidence")
    if isinstance(evidence, dict):
        rule = str(evidence.get("rule") or "").strip()
        if rule:
            return rule
    sid = str(suggestion.get("id") or "")
    # R1_high_drift_low_importance → high_drift_low_importance
    if "_" in sid:
        parts = sid.split("_", 1)
        if parts[0].startswith("R") and parts[0][1:].isdigit():
            return parts[1]
    return sid or "unknown"


def _rule_id(suggestion: dict[str, Any]) -> str:
    sid = str(suggestion.get("id") or "").strip()
    if "__" in sid:
        return sid.split("__", 1)[0]
    return sid


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return sum(vals) / len(vals)


def _feat_metric(item: Any, *keys: str) -> float | None:
    if not isinstance(item, dict):
        return None
    for key in keys:
        f = _f(item.get(key))
        if f is not None:
            return f
    return None


def aggregate_feature_stats(feats: list[Any]) -> dict[str, float | None]:
    risks = [v for v in (_feat_metric(f, "risk_score") for f in feats) if v is not None]
    drifts = [v for v in (_feat_metric(f, "drift") for f in feats) if v is not None]
    kss = [v for v in (_feat_metric(f, "ks_statistic") for f in feats) if v is not None]
    ranks = [v for v in (_feat_metric(f, "rank_gain") for f in feats) if v is not None]
    return {
        "avg_risk": _mean(risks),
        "avg_drift": _mean(drifts),
        "avg_ks": _mean(kss),
        "avg_rank_gain": _mean(ranks),
        "feature_count": float(len(feats)),
    }


def experiment_rank_score(
    feats: list[Any],
    *,
    parent_evidence_score: float,
) -> float:
    """Aggregate evidence for ranking (higher = more valuable first).

    Formula (components clipped to ~0–100, then blended with parent score)::

        agg = 0.40 * avg_risk
            + 0.30 * min(avg_drift, 1) * 100
            + 0.20 * min(avg_ks, 1) * 100
            + 0.10 * min(avg_rank_gain / 2, 100)   # importance penalty
        rank = 0.45 * parent_evidence_score + 0.55 * agg

    Missing metrics contribute 0 for that term. ``avg_rank_gain`` treats higher
    ranks (less important) as a larger removal/review penalty.
    """
    stats = aggregate_feature_stats(feats)
    avg_risk = float(stats["avg_risk"] or 0.0)
    avg_drift = float(stats["avg_drift"] or 0.0)
    avg_ks = float(stats["avg_ks"] or 0.0)
    avg_rank = float(stats["avg_rank_gain"] or 0.0)

    agg = (
        0.40 * min(max(avg_risk, 0.0), 100.0)
        + 0.30 * min(max(avg_drift, 0.0), 1.0) * 100.0
        + 0.20 * min(max(avg_ks, 0.0), 1.0) * 100.0
        + 0.10 * min(max(avg_rank, 0.0) / 2.0, 100.0)
    )
    parent = max(0.0, min(100.0, float(parent_evidence_score)))
    if not feats:
        return parent
    return 0.45 * parent + 0.55 * agg


def _parent_evidence_score(suggestion: dict[str, Any]) -> float:
    score = _f(suggestion.get("evidence_score"))
    if score is not None:
        if score <= 1.0 and suggestion.get("confidence") is not None:
            unit = unit_from_evidence_score(suggestion.get("confidence"))
            return (unit or 0.0) * 100.0
        return float(score)
    unit = unit_from_evidence_score(suggestion.get("confidence"))
    return (unit or 0.0) * 100.0


def _top_contributors(feats: list[Any], limit: int) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in feats:
        if not isinstance(item, dict):
            name = str(item).strip()
            if name:
                scored.append((0.0, {"feature": name}))
            continue
        risk = _feat_metric(item, "risk_score") or 0.0
        drift = _feat_metric(item, "drift") or 0.0
        ks = _feat_metric(item, "ks_statistic") or 0.0
        # Prefer risk, then drift/KS as tie-breakers.
        key = risk * 1.0 + drift * 20.0 + ks * 10.0
        scored.append((key, dict(item)))
    scored.sort(key=lambda t: (-t[0], str(t[1].get("feature") or "")))
    return [obj for _k, obj in scored[: max(0, int(limit))]]


def _thresholds_from_evidence(evidence: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(evidence, dict):
        return {}
    th = evidence.get("thresholds")
    if isinstance(th, dict):
        return dict(th)
    # Some rules store a single threshold key.
    out: dict[str, Any] = {}
    for key in ("threshold_pp", "large_ks", "small_mean_drift_pct"):
        if key in evidence:
            out[key] = evidence[key]
    return out


def build_hypothesis(rule_name: str, family: str | None) -> str:
    tmpl = _HYPOTHESIS_BY_RULE.get(rule_name)
    fam = (family or "model").strip() or "model"
    if tmpl:
        try:
            return tmpl.format(family=fam)
        except (KeyError, ValueError):
            return tmpl
    if family:
        return (
            f"Reviewing these {fam} features under rule '{rule_name}' may clarify "
            "whether the matched signal is actionable."
        )
    return (
        f"Acting on rule '{rule_name}' may address the diagnostics or structural "
        "signal without requiring a feature ablation."
    )


def build_title(rule_name: str, family: str | None, fallback: str) -> str:
    tmpl = _TITLE_BY_RULE.get(rule_name)
    if tmpl and family:
        try:
            return tmpl.format(family=family)
        except (KeyError, ValueError):
            pass
    if tmpl and "{family}" not in tmpl:
        return tmpl
    return fallback


def _reason_bullets_for_family(
    parent: dict[str, Any],
    *,
    family: str | None,
    count: int,
) -> list[str]:
    bullets: list[str] = []
    raw = parent.get("reason_bullets")
    if isinstance(raw, list):
        for b in raw:
            text = str(b).strip()
            if not text:
                continue
            # Drop parent-wide "N features matched" — replaced below.
            if "features matched" in text.lower():
                continue
            bullets.append(text)
    if family:
        bullets.append(f"Family: {family}")
    if count:
        bullets.append(f"{count} features matched")
    if not bullets:
        reason = str(parent.get("reason") or "").strip()
        if reason:
            bullets.append(reason)
    return bullets


def _enrich_evidence(
    parent: dict[str, Any],
    *,
    family: str | None,
    feats: list[Any],
    top_n: int,
    rank_score: float,
) -> dict[str, Any]:
    parent_ev = parent.get("evidence") if isinstance(parent.get("evidence"), dict) else {}
    rule_name = _rule_name(parent)
    rule_id = _rule_id(parent)
    names = feature_names(feats)
    stats = aggregate_feature_stats(feats)
    highest_name, highest_risk = _highest_risk_feature(feats)
    evidence: dict[str, Any] = {
        "rule": rule_name,
        "rule_id": rule_id,
        "rule_name": rule_name,
        "family": family,
        "feature_count": len(names),
        "matched_features": names,
        "top_contributors": _top_contributors(feats, top_n),
        "thresholds": _thresholds_from_evidence(parent_ev),
        "aggregate": {
            "avg_risk": round(stats["avg_risk"], 4) if stats["avg_risk"] is not None else None,
            "avg_drift": round(stats["avg_drift"], 4) if stats["avg_drift"] is not None else None,
            "avg_ks": round(stats["avg_ks"], 4) if stats["avg_ks"] is not None else None,
            "avg_rank_gain": (
                round(stats["avg_rank_gain"], 4)
                if stats["avg_rank_gain"] is not None
                else None
            ),
            "rank_score": round(float(rank_score), 2),
            "highest_risk_feature": highest_name,
            "highest_risk_score": (
                round(highest_risk, 4) if highest_risk is not None else None
            ),
        },
    }
    # Preserve useful parent evidence keys (diagnostics fields, max_*, etc.).
    for key, val in parent_ev.items():
        if key in evidence:
            continue
        if key == "samples":
            # Filter samples to this family when possible.
            if isinstance(val, list) and family:
                filtered = []
                for s in val:
                    if not isinstance(s, dict):
                        continue
                    fname = str(s.get("feature") or "")
                    if resolve_feature_family(fname) == family or fname in names:
                        filtered.append(s)
                evidence["samples"] = filtered[:10]
            else:
                evidence["samples"] = val
            continue
        evidence[key] = val
    if "samples" not in evidence:
        evidence["samples"] = [
            {"feature": n} if not isinstance(f, dict) else dict(f)
            for n, f in zip(names[:10], feats[:10])
        ]
    return evidence


def split_suggestion_by_family(
    suggestion: dict[str, Any],
    *,
    family_by_name: dict[str, str] | None = None,
    top_contributors: int = 5,
    th: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Split one raw rule suggestion into per-family research experiments."""
    thresholds = th or merge_thresholds()
    feats = suggestion.get("affected_features")
    if not isinstance(feats, list):
        feats = []
    parent_score = _parent_evidence_score(suggestion)
    rule_name = _rule_name(suggestion)
    base_id = _rule_id(suggestion)

    if not feats:
        # Diagnostics / addition hints: single experiment, no family split.
        hyp = build_hypothesis(rule_name, None)
        title = build_title(rule_name, None, str(suggestion.get("title") or base_id))
        score = int(round(parent_score))
        unit = score / 100.0
        cat = suggestion.get("category")
        scope = experiment_scope(str(cat) if cat is not None else None, None)
        return [
            {
                "id": base_id,
                "category": cat,
                "title": title,
                "family": None,
                "reason": " · ".join(
                    _reason_bullets_for_family(suggestion, family=None, count=0)
                ),
                "reason_bullets": _reason_bullets_for_family(
                    suggestion, family=None, count=0
                ),
                "hypothesis": hyp,
                "expected_experiment": build_expected_experiment(rule_name, None),
                "suggested_next_steps": build_suggested_next_steps(
                    category=str(cat) if cat is not None else None,
                    family=None,
                    scope=scope,
                ),
                "estimated_effort": estimate_effort(
                    category=str(cat) if cat is not None else None,
                    feature_count=0,
                    family=None,
                ),
                "experiment_scope": scope,
                "status": DEFAULT_EXPERIMENT_STATUS,
                "evidence": _enrich_evidence(
                    suggestion,
                    family=None,
                    feats=[],
                    top_n=top_contributors,
                    rank_score=parent_score,
                ),
                "evidence_score": score,
                "confidence": round(unit, 3),
                "rank_score": round(parent_score, 2),
                "priority": _priority_from_unit(unit, thresholds),
                "affected_features": [],
            }
        ]

    grouped = group_features_by_family(feats, family_by_name=family_by_name)
    out: list[dict[str, Any]] = []
    for family, fam_feats in grouped.items():
        rank = experiment_rank_score(fam_feats, parent_evidence_score=parent_score)
        score = int(round(max(0.0, min(100.0, rank))))
        unit = score / 100.0
        count = len(fam_feats)
        hyp = build_hypothesis(rule_name, family)
        title = build_title(
            rule_name,
            family,
            f"Review {family} Features",
        )
        bullets = _reason_bullets_for_family(
            suggestion, family=family, count=count
        )
        exp_id = f"{base_id}__{family}"
        cat = suggestion.get("category")
        scope = experiment_scope(str(cat) if cat is not None else None, family)
        out.append(
            {
                "id": exp_id,
                "category": cat,
                "title": title,
                "family": family,
                "reason": " · ".join(bullets),
                "reason_bullets": bullets,
                "hypothesis": hyp,
                "expected_experiment": build_expected_experiment(rule_name, family),
                "suggested_next_steps": build_suggested_next_steps(
                    category=str(cat) if cat is not None else None,
                    family=family,
                    scope=scope,
                ),
                "estimated_effort": estimate_effort(
                    category=str(cat) if cat is not None else None,
                    feature_count=count,
                    family=family,
                ),
                "experiment_scope": scope,
                "status": DEFAULT_EXPERIMENT_STATUS,
                "evidence": _enrich_evidence(
                    suggestion,
                    family=family,
                    feats=fam_feats,
                    top_n=top_contributors,
                    rank_score=rank,
                ),
                "evidence_score": score,
                "confidence": round(unit, 3),
                "rank_score": round(rank, 2),
                "priority": _priority_from_unit(unit, thresholds),
                "affected_features": list(fam_feats),
            }
        )
    return out


def _finding_from_experiment(exp: dict[str, Any]) -> dict[str, Any]:
    """Build a findings[] entry from a single-rule family experiment."""
    rule_name = _rule_name(exp)
    rule_id = _rule_id(exp)
    feats = exp.get("affected_features") if isinstance(exp.get("affected_features"), list) else []
    return {
        "rule": rule_name,
        "rule_id": rule_id,
        "rule_name": rule_name,
        "recommendation": finding_label(rule_name, str(exp.get("title") or "")),
        "title": str(exp.get("title") or ""),
        "category": exp.get("category"),
        "hypothesis": str(exp.get("hypothesis") or ""),
        "expected_experiment": str(exp.get("expected_experiment") or ""),
        "evidence_score": exp.get("evidence_score"),
        "rank_score": exp.get("rank_score"),
        "feature_count": len(feats),
        "matched_features": feature_names(feats),
    }


def _union_features(groups: list[list[Any]]) -> list[Any]:
    """Merge feature lists by name, keeping the richest dict."""
    by_name: dict[str, Any] = {}
    order: list[str] = []
    for feats in groups:
        for item in feats:
            if isinstance(item, dict):
                name = str(item.get("feature") or "").strip()
                if not name:
                    continue
                if name not in by_name:
                    order.append(name)
                    by_name[name] = dict(item)
                else:
                    prev = by_name[name]
                    if not isinstance(prev, dict):
                        by_name[name] = dict(item)
                        continue
                    merged = dict(prev)
                    for k, v in item.items():
                        if k == "feature":
                            continue
                        if merged.get(k) is None and v is not None:
                            merged[k] = v
                    by_name[name] = merged
            else:
                name = str(item).strip()
                if not name:
                    continue
                if name not in by_name:
                    order.append(name)
                    by_name[name] = {"feature": name}
    return [by_name[n] for n in order]


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        text = str(v or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _merge_key(exp: dict[str, Any]) -> str | None:
    """Return merge bucket key for feature-family experiments; None = keep separate."""
    scope = str(exp.get("experiment_scope") or "").strip().lower()
    family = exp.get("family")
    if family is None or not str(family).strip():
        return None
    if scope == "model":
        # Model-scope with a family still merges by family+scope.
        return f"model__{str(family).strip()}"
    return f"feature__{str(family).strip()}"


def merge_experiments_by_family(
    experiments: list[dict[str, Any]],
    *,
    top_contributors: int = 5,
    th: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Merge same-family (and scope) rule findings into one experiment with findings[].

    Avoids duplicate family experiments such as \"Review Time Features\" and
    \"Review Time Shape Shift\" — one \"Time Features\" experiment with multiple
    recommendation findings. Model-level (no family) experiments stay separate.
    """
    thresholds = th or merge_thresholds()
    buckets: dict[str, list[dict[str, Any]]] = {}
    singles: list[dict[str, Any]] = []
    bucket_order: list[str] = []

    for exp in experiments:
        if not isinstance(exp, dict):
            continue
        key = _merge_key(exp)
        if key is None:
            row = dict(exp)
            if not isinstance(row.get("findings"), list) or not row["findings"]:
                row["findings"] = [_finding_from_experiment(row)]
            if not isinstance(row.get("recommendations"), list) or not row["recommendations"]:
                row["recommendations"] = [
                    f.get("recommendation")
                    for f in row["findings"]
                    if isinstance(f, dict) and f.get("recommendation")
                ]
            singles.append(row)
            continue
        if key not in buckets:
            buckets[key] = []
            bucket_order.append(key)
        buckets[key].append(exp)

    merged: list[dict[str, Any]] = []
    for key in bucket_order:
        group = buckets[key]
        if len(group) == 1:
            row = dict(group[0])
            finding = _finding_from_experiment(row)
            row["findings"] = [finding]
            row["recommendations"] = [finding["recommendation"]]
            # Stable internal id for family bucket (even with one finding).
            family = str(row.get("family") or "").strip()
            scope = str(row.get("experiment_scope") or "feature")
            row["id"] = f"{scope}__{family}" if family else row.get("id")
            merged.append(row)
            continue

        # Multi-rule merge
        family = str(group[0].get("family") or "").strip()
        scope = str(group[0].get("experiment_scope") or "feature")
        findings = [_finding_from_experiment(g) for g in group]
        # Highest rank_score first in findings
        findings.sort(
            key=lambda f: (-float(f.get("rank_score") or f.get("evidence_score") or 0), str(f.get("rule_id") or ""))
        )
        recommendations = [str(f.get("recommendation") or "") for f in findings if f.get("recommendation")]

        all_feats = _union_features(
            [
                g.get("affected_features")
                if isinstance(g.get("affected_features"), list)
                else []
                for g in group
            ]
        )
        parent_scores = [_parent_evidence_score(g) for g in group]
        parent_score = max(parent_scores) if parent_scores else 0.0
        rank = experiment_rank_score(all_feats, parent_evidence_score=parent_score)
        score = int(round(max(0.0, min(100.0, rank))))
        unit = score / 100.0

        categories = _unique_strings([str(g.get("category") or "") for g in group])
        # Prefer Feature Removal if present, else first by findings order.
        if "Feature Removal" in categories and len(categories) > 1:
            category = "Feature Removal"
        elif "Data Collection" in categories and len(categories) > 1:
            category = "Data Collection"
        elif "Model Refresh" in categories and len(categories) > 1:
            category = "Model Refresh"
        else:
            category = categories[0] if categories else "Feature Review"

        hyps = _unique_strings([str(g.get("hypothesis") or "") for g in group])
        expecteds = _unique_strings([str(g.get("expected_experiment") or "") for g in group])
        steps: list[str] = []
        for g in group:
            raw_steps = g.get("suggested_next_steps")
            if isinstance(raw_steps, list):
                steps.extend(str(s) for s in raw_steps)
        steps = _unique_strings(steps)

        # Reason: recommendations + family + count
        reason_bullets = [f"Recommendation: {rec}" for rec in recommendations]
        reason_bullets.append(f"Family: {family}")
        reason_bullets.append(f"{len(all_feats)} features matched")
        reason_bullets.append(f"{len(findings)} findings merged")

        # Build evidence from richest parent + union features
        primary = max(group, key=lambda g: float(g.get("rank_score") or 0))
        evidence = _enrich_evidence(
            primary,
            family=family,
            feats=all_feats,
            top_n=top_contributors,
            rank_score=rank,
        )
        evidence["findings"] = findings
        evidence["merged_rule_ids"] = [f.get("rule_id") for f in findings]
        evidence["merged_rules"] = [f.get("rule") for f in findings]

        title = f"{family} Features" if family else str(primary.get("title") or key)
        hypothesis = " ".join(hyps) if hyps else build_hypothesis(_rule_name(primary), family)
        expected = " ".join(expecteds) if expecteds else build_expected_experiment(_rule_name(primary), family)

        merged.append(
            {
                "id": f"{scope}__{family}" if family else key,
                "category": category,
                "title": title,
                "family": family or None,
                "reason": " · ".join(reason_bullets),
                "reason_bullets": reason_bullets,
                "hypothesis": hypothesis,
                "expected_experiment": expected,
                "suggested_next_steps": steps
                or build_suggested_next_steps(
                    category=category, family=family or None, scope=scope
                ),
                "estimated_effort": estimate_effort(
                    category=category,
                    feature_count=len(all_feats),
                    family=family or None,
                ),
                "experiment_scope": scope,
                "status": DEFAULT_EXPERIMENT_STATUS,
                "evidence": evidence,
                "evidence_score": score,
                "confidence": round(unit, 3),
                "rank_score": round(rank, 2),
                "priority": _priority_from_unit(unit, thresholds),
                "affected_features": all_feats,
                "findings": findings,
                "recommendations": recommendations,
            }
        )

    return merged + singles


def stamp_generation_meta(
    experiments: list[dict[str, Any]],
    *,
    model_name: str | None = None,
    generated_at: str | None = None,
    planner_version: str | None = None,
) -> list[dict[str, Any]]:
    """Attach created_from / generated_at / planner_version to each experiment."""
    created = str(model_name or "").strip() or None
    at = str(generated_at or "").strip() or None
    version = str(planner_version or PLANNER_VERSION).strip() or PLANNER_VERSION
    out: list[dict[str, Any]] = []
    for exp in experiments:
        row = dict(exp)
        row["created_from"] = created
        row["generated_at"] = at
        row["planner_version"] = version
        out.append(row)
    return out


def assign_stable_experiment_ids(
    experiments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign EXP-001… after final rank/cap order. Keeps internal ``id``."""
    out: list[dict[str, Any]] = []
    for i, exp in enumerate(experiments, start=1):
        row = dict(exp)
        row["experiment_id"] = format_experiment_id(i)
        # Ensure defaults if older callers skipped split enrichment.
        if not row.get("status"):
            row["status"] = DEFAULT_EXPERIMENT_STATUS
        if not row.get("estimated_effort"):
            feats = row.get("affected_features")
            n = len(feats) if isinstance(feats, list) else 0
            row["estimated_effort"] = estimate_effort(
                category=str(row.get("category") or "") or None,
                feature_count=n,
                family=str(row.get("family") or "") or None,
            )
        if not row.get("experiment_scope"):
            row["experiment_scope"] = experiment_scope(
                str(row.get("category") or "") or None,
                str(row.get("family") or "") or None,
            )
        if not str(row.get("expected_experiment") or "").strip():
            row["expected_experiment"] = build_expected_experiment(
                _rule_name(row),
                str(row.get("family") or "") or None,
            )
        steps = row.get("suggested_next_steps")
        if not isinstance(steps, list) or not steps:
            row["suggested_next_steps"] = build_suggested_next_steps(
                category=str(row.get("category") or "") or None,
                family=str(row.get("family") or "") or None,
                scope=str(row.get("experiment_scope") or "feature"),
            )
        if not isinstance(row.get("findings"), list) or not row["findings"]:
            row["findings"] = [_finding_from_experiment(row)]
        if not isinstance(row.get("recommendations"), list) or not row["recommendations"]:
            row["recommendations"] = [
                str(f.get("recommendation") or "")
                for f in row["findings"]
                if isinstance(f, dict) and f.get("recommendation")
            ]
        out.append(row)
    return out


def refine_to_experiments(
    suggestions: list[dict[str, Any]],
    *,
    thresholds: dict[str, Any] | None = None,
    family_by_name: dict[str, str] | None = None,
    model_name: str | None = None,
    generated_at: str | None = None,
    planner_version: str | None = None,
) -> list[dict[str, Any]]:
    """Split, merge-by-family, rank, and cap rule suggestions into research experiments."""
    th = merge_thresholds(thresholds)
    max_exp = int(th.get("max_experiments") or 10)
    top_n = int(th.get("top_contributors") or 5)

    experiments: list[dict[str, Any]] = []
    for sug in suggestions:
        if not isinstance(sug, dict):
            continue
        experiments.extend(
            split_suggestion_by_family(
                sug,
                family_by_name=family_by_name,
                top_contributors=top_n,
                th=th,
            )
        )

    # Merge same-family findings (presentation only — no new rules).
    experiments = merge_experiments_by_family(
        experiments, top_contributors=top_n, th=th
    )

    # Deduplicate by id (keep highest rank_score).
    best: dict[str, dict[str, Any]] = {}
    for exp in experiments:
        eid = str(exp.get("id") or "")
        if not eid:
            continue
        prev = best.get(eid)
        if prev is None or float(exp.get("rank_score") or 0) > float(
            prev.get("rank_score") or 0
        ):
            best[eid] = exp
    unique = list(best.values())

    unique.sort(
        key=lambda s: (
            -float(s.get("rank_score") or s.get("evidence_score") or 0),
            str(s.get("id") or ""),
        )
    )
    if max_exp > 0:
        unique = unique[:max_exp]
    unique = assign_stable_experiment_ids(unique)
    return stamp_generation_meta(
        unique,
        model_name=model_name,
        generated_at=generated_at,
        planner_version=planner_version,
    )
