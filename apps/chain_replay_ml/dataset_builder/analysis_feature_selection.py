"""Feature Selection Strategy — how a Final Feature Dataset is built.

Strategies (mutually exclusive experiments):

* ``hca_corr_perm`` — Corr → HCA families → representative policy (Top 1/2/3/N)
* ``corr_perm`` — Correlation filter → Permutation ranking → keep above threshold
* ``perm_only`` — Permutation ranking only (research)
* ``corr_only`` — Correlation filter only (research)

Every frozen discovery_bundle / experiment_result / champion should embed
``feature_selection`` + ``final_feature_dataset`` so models remain reproducible.
"""
from __future__ import annotations

import json
from typing import Any

from .analysis_lab_store import _AnalysisDb, _now_iso

STRATEGY_HCA = "hca_corr_perm"
STRATEGY_CORR_PERM = "corr_perm"
STRATEGY_PERM_ONLY = "perm_only"
STRATEGY_CORR_ONLY = "corr_only"

STRATEGY_LABELS: dict[str, str] = {
    STRATEGY_HCA: "HCA + Correlation + Permutation",
    STRATEGY_CORR_PERM: "Correlation + Permutation Only",
    STRATEGY_PERM_ONLY: "Permutation Only (Research)",
    STRATEGY_CORR_ONLY: "Correlation Only (Research)",
}

STRATEGY_SHORT: dict[str, str] = {
    STRATEGY_HCA: "HCA",
    STRATEGY_CORR_PERM: "Corr+Perm",
    STRATEGY_PERM_ONLY: "Perm Only",
    STRATEGY_CORR_ONLY: "Corr Only",
}

POLICY_TOP_1 = "top_1"
POLICY_TOP_2 = "top_2"
POLICY_TOP_3 = "top_3"
POLICY_TOP_N = "top_n"

POLICY_LABELS: dict[str, str] = {
    POLICY_TOP_1: "Top 1",
    POLICY_TOP_2: "Top 2",
    POLICY_TOP_3: "Top 3",
    POLICY_TOP_N: "Top N",
}

DEFAULT_CORR_THRESHOLD = 0.95
DEFAULT_PERM_THRESHOLD = 0.001
DEFAULT_TOP_N = 1

ALL_STRATEGIES = (
    STRATEGY_HCA,
    STRATEGY_CORR_PERM,
    STRATEGY_PERM_ONLY,
    STRATEGY_CORR_ONLY,
)


def normalize_strategy(strategy: str | None) -> str:
    s = str(strategy or STRATEGY_HCA).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "hca": STRATEGY_HCA,
        "hca_corr_perm": STRATEGY_HCA,
        "hca_correlation_permutation": STRATEGY_HCA,
        "corr_perm": STRATEGY_CORR_PERM,
        "correlation_permutation": STRATEGY_CORR_PERM,
        "correlation_perm": STRATEGY_CORR_PERM,
        "perm_only": STRATEGY_PERM_ONLY,
        "permutation_only": STRATEGY_PERM_ONLY,
        "perm": STRATEGY_PERM_ONLY,
        "corr_only": STRATEGY_CORR_ONLY,
        "correlation_only": STRATEGY_CORR_ONLY,
        "corr": STRATEGY_CORR_ONLY,
    }
    return aliases.get(s, s if s in ALL_STRATEGIES else STRATEGY_HCA)


def normalize_policy(
    policy: str | None, *, top_n: int | None = None
) -> tuple[str, int]:
    p = str(policy or POLICY_TOP_1).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "top1": POLICY_TOP_1,
        "top_1": POLICY_TOP_1,
        "1": POLICY_TOP_1,
        "top2": POLICY_TOP_2,
        "top_2": POLICY_TOP_2,
        "2": POLICY_TOP_2,
        "top3": POLICY_TOP_3,
        "top_3": POLICY_TOP_3,
        "3": POLICY_TOP_3,
        "topn": POLICY_TOP_N,
        "top_n": POLICY_TOP_N,
        "n": POLICY_TOP_N,
    }
    p = aliases.get(p, p if p in POLICY_LABELS else POLICY_TOP_1)
    if p == POLICY_TOP_1:
        n = 1
    elif p == POLICY_TOP_2:
        n = 2
    elif p == POLICY_TOP_3:
        n = 3
    else:
        n = max(int(top_n or DEFAULT_TOP_N), 1)
        if n in (1, 2, 3):
            p = {1: POLICY_TOP_1, 2: POLICY_TOP_2, 3: POLICY_TOP_3}[n]
    return p, n


def build_selection_config(
    strategy: str = STRATEGY_HCA,
    *,
    representative_policy: str = POLICY_TOP_1,
    top_n: int | None = None,
    correlation_threshold: float = DEFAULT_CORR_THRESHOLD,
    permutation_threshold: float = DEFAULT_PERM_THRESHOLD,
    min_family_size: int = 2,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalized, serializable Feature Selection config."""
    sid = normalize_strategy(strategy)
    policy, n = normalize_policy(representative_policy, top_n=top_n)
    cfg: dict[str, Any] = {
        "strategy": sid,
        "strategy_label": STRATEGY_LABELS.get(sid, sid),
        "strategy_short": STRATEGY_SHORT.get(sid, sid),
        "version": 1,
        "correlation_threshold": float(correlation_threshold),
        "permutation_threshold": float(permutation_threshold),
        "min_family_size": int(min_family_size),
    }
    if sid == STRATEGY_HCA:
        cfg["representative_policy"] = policy
        cfg["representative_policy_label"] = POLICY_LABELS.get(policy, policy)
        cfg["top_n"] = int(n)
    if extra:
        cfg.update(dict(extra))
    return cfg


def required_modules_for_strategy(strategy: str) -> tuple[str, ...]:
    sid = normalize_strategy(strategy)
    if sid == STRATEGY_HCA:
        return (
            "correlation",
            "hca",
            "mutual_information",
            "permutation",
            "feature_scorecard",
        )
    if sid == STRATEGY_CORR_PERM:
        return ("correlation", "permutation")
    if sid == STRATEGY_PERM_ONLY:
        return ("permutation",)
    if sid == STRATEGY_CORR_ONLY:
        return ("correlation",)
    return ("correlation", "hca", "permutation", "feature_scorecard")


def format_selection_summary(
    cfg: dict[str, Any] | None, *, n_features: int | None = None
) -> str:
    """Human-readable Feature Selection block for Experiment Details / Champion."""
    c = dict(cfg or {})
    sid = normalize_strategy(c.get("strategy"))
    lines = [
        "Feature Selection",
        f"  Strategy              {c.get('strategy_label') or STRATEGY_LABELS.get(sid, sid)}",
    ]
    if sid == STRATEGY_HCA:
        lines.append(
            "  Representative Policy "
            + str(
                c.get("representative_policy_label")
                or POLICY_LABELS.get(
                    str(c.get("representative_policy") or ""),
                    c.get("representative_policy") or "Top 1",
                )
            )
        )
        if c.get("n_families") is not None:
            lines.append(f"  Families              {c.get('n_families')}")
        if c.get("top_n") is not None:
            lines.append(f"  Top N                 {c.get('top_n')}")
    else:
        if c.get("correlation_threshold") is not None and sid in (
            STRATEGY_CORR_PERM,
            STRATEGY_CORR_ONLY,
        ):
            lines.append(
                f"  Correlation Threshold {float(c['correlation_threshold']):.3f}"
            )
        if c.get("permutation_threshold") is not None and sid in (
            STRATEGY_CORR_PERM,
            STRATEGY_PERM_ONLY,
        ):
            lines.append(
                f"  Permutation Threshold {float(c['permutation_threshold']):.6g}"
            )
    nf = n_features if n_features is not None else c.get("n_selected_features")
    if nf is not None:
        lines.append(f"  Selected Features     {nf}")
    return "\n".join(lines)


# Overview display labels (slightly shorter than STRATEGY_LABELS for Model Registry).
_OVERVIEW_STRATEGY_LABELS: dict[str, str] = {
    STRATEGY_HCA: "HCA + Correlation + Permutation",
    STRATEGY_CORR_PERM: "Correlation + Permutation",
    STRATEGY_PERM_ONLY: "Permutation Only",
    STRATEGY_CORR_ONLY: "Correlation Only",
}


def _short_hash(value: Any, *, keep: int = 8) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= keep:
        return text
    return f"{text[:keep]}..."


def build_feature_selection_lineage(
    preview: dict[str, Any] | None,
    *,
    source: str = "analysis",
    run_id: str | None = None,
    analysis_dataset: str | None = None,
    discovery_bundle_id: str | None = None,
) -> dict[str, Any]:
    """Serializable Feature Selection identity for Model Builder / Model Registry.

    Built from a ``preview_selection`` / ``build_final_feature_dataset`` result and
    frozen onto production packages so Overview can tell the creation story.
    """
    prev = dict(preview or {})
    ds = dict(prev.get("dataset") or {}) if isinstance(prev.get("dataset"), dict) else {}
    cfg = dict(prev.get("feature_selection") or ds.get("feature_selection") or {})
    if not cfg.get("strategy") and prev.get("strategy"):
        cfg = {
            **cfg,
            "strategy": prev.get("strategy"),
            "strategy_label": prev.get("strategy_label"),
            "representative_policy": prev.get("representative_policy"),
            "representative_policy_label": prev.get("representative_policy_label"),
            "top_n": prev.get("top_n"),
            "correlation_threshold": prev.get("correlation_threshold"),
            "permutation_threshold": prev.get("permutation_threshold"),
            "n_families": prev.get("n_families"),
            "n_after_correlation": prev.get("n_after_correlation"),
            "n_after_permutation": prev.get("n_after_permutation"),
            "n_input_features": prev.get("n_input_features"),
            "n_selected_features": prev.get("n_features") or prev.get("count"),
        }
    sid = normalize_strategy(cfg.get("strategy") or prev.get("strategy"))
    features = [
        str(f).strip()
        for f in (prev.get("features") or ds.get("features") or [])
        if str(f).strip()
    ]
    feature_hash = str(
        prev.get("hash")
        or ds.get("hash")
        or (ds.get("feature_set") or {}).get("hash")
        or ""
    ).strip()
    if not feature_hash and features:
        from .analysis_experiments import features_fingerprint

        feature_hash = features_fingerprint(features)

    n_input = prev.get("n_input_features")
    if n_input is None:
        n_input = cfg.get("n_input_features")
    if n_input is None:
        n_input = ds.get("n_input_features")
    n_after_corr = prev.get("n_after_correlation")
    if n_after_corr is None:
        n_after_corr = cfg.get("n_after_correlation")
    n_families = prev.get("n_families")
    if n_families is None:
        n_families = cfg.get("n_families")
    n_selected = prev.get("n_features")
    if n_selected is None:
        n_selected = prev.get("count")
    if n_selected is None:
        n_selected = cfg.get("n_selected_features")
    if n_selected is None:
        n_selected = len(features)

    strategy_label = (
        _OVERVIEW_STRATEGY_LABELS.get(sid)
        or cfg.get("strategy_label")
        or STRATEGY_LABELS.get(sid, sid)
    )
    lineage: dict[str, Any] = {
        "version": 1,
        "source": str(source or "analysis").strip() or "analysis",
        "strategy": sid,
        "strategy_label": strategy_label,
        "correlation_threshold": cfg.get("correlation_threshold"),
        "permutation_threshold": cfg.get("permutation_threshold"),
        "n_input_features": n_input,
        "n_after_correlation": n_after_corr,
        "n_families": n_families,
        "n_selected_features": n_selected,
        "feature_set_hash": feature_hash,
        "features": features,
        "run_id": str(run_id or "").strip() or None,
        "analysis_dataset": str(analysis_dataset or "").strip() or None,
        "discovery_bundle_id": str(discovery_bundle_id or "").strip() or None,
        "pipeline": list(prev.get("pipeline") or ds.get("pipeline") or []),
        "feature_selection": cfg or None,
    }
    if sid == STRATEGY_HCA:
        lineage["representative_policy"] = cfg.get("representative_policy") or prev.get(
            "representative_policy"
        )
        lineage["representative_policy_label"] = (
            cfg.get("representative_policy_label")
            or prev.get("representative_policy_label")
            or POLICY_LABELS.get(str(lineage.get("representative_policy") or ""), None)
        )
        lineage["top_n"] = cfg.get("top_n") or prev.get("top_n")
    return lineage


def feature_selection_overview_rows(
    lineage: dict[str, Any] | None,
) -> list[tuple[str, Any]]:
    """Adaptive Overview rows — HCA-only fields omitted for corr/perm strategies."""
    lin = dict(lineage or {})
    if not lin:
        return []
    sid = normalize_strategy(lin.get("strategy"))
    source = str(lin.get("source") or "analysis").strip()
    source_label = {
        "analysis": "Analysis",
        "research": "Analysis",
        "feature_selection": "Analysis",
        "model_builder": "Model Builder",
        "manual": "Manual",
    }.get(source.lower(), source.title() if source else "Analysis")

    rows: list[tuple[str, Any]] = [
        ("Source", source_label),
        (
            "Selection Strategy",
            lin.get("strategy_label")
            or _OVERVIEW_STRATEGY_LABELS.get(sid)
            or STRATEGY_LABELS.get(sid, sid),
        ),
    ]
    if sid == STRATEGY_HCA:
        pol = lin.get("representative_policy_label") or POLICY_LABELS.get(
            str(lin.get("representative_policy") or ""), None
        )
        if pol:
            rows.append(("Representative Policy", pol))

    if sid in (STRATEGY_HCA, STRATEGY_CORR_PERM, STRATEGY_CORR_ONLY):
        if lin.get("correlation_threshold") is not None:
            rows.append(
                ("Correlation Threshold", f"{float(lin['correlation_threshold']):.2f}")
            )
    if sid in (STRATEGY_HCA, STRATEGY_CORR_PERM, STRATEGY_PERM_ONLY):
        if lin.get("permutation_threshold") is not None:
            rows.append(
                ("Permutation Threshold", f"{float(lin['permutation_threshold']):.6g}")
            )

    if lin.get("n_input_features") is not None:
        rows.append(("Original Features", lin.get("n_input_features")))
    if sid in (STRATEGY_HCA, STRATEGY_CORR_PERM, STRATEGY_CORR_ONLY):
        if lin.get("n_after_correlation") is not None:
            rows.append(("After Correlation", lin.get("n_after_correlation")))
    if sid == STRATEGY_HCA and lin.get("n_families") is not None:
        rows.append(("HCA Families", lin.get("n_families")))
    if lin.get("n_selected_features") is not None:
        rows.append(("Selected Features", lin.get("n_selected_features")))

    fh = _short_hash(lin.get("feature_set_hash"))
    if fh:
        rows.append(("Feature Set Hash", fh))
    if lin.get("discovery_bundle_id"):
        rows.append(("Discovery Bundle", lin.get("discovery_bundle_id")))
    if lin.get("analysis_dataset"):
        rows.append(("Analysis Dataset", lin.get("analysis_dataset")))
    return rows


def extract_feature_selection_lineage(
    doc: dict[str, Any] | None,
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> dict[str, Any] | None:
    """Pull lineage from a model detail / config document."""
    if not isinstance(doc, dict):
        return None
    if _depth > 8:
        return None
    seen = _seen if _seen is not None else set()
    doc_id = id(doc)
    if doc_id in seen:
        return None
    seen.add(doc_id)
    for key in (
        "analysis_feature_selection",
        "feature_selection_lineage",
        "final_feature_dataset",
    ):
        raw = doc.get(key)
        if isinstance(raw, dict) and (
            raw.get("strategy") or raw.get("features") or raw.get("feature_selection")
        ):
            if key == "final_feature_dataset" and not raw.get("strategy"):
                return build_feature_selection_lineage(raw)
            if raw.get("strategy") or raw.get("strategy_label"):
                return dict(raw)
            return build_feature_selection_lineage(raw)
    # Only recurse into real nested dicts — never synthesize a fresh ``{}``
    # (that caused infinite recursion when config was missing).
    for key in ("config", "training_config", "feature_selection"):
        nested_doc = doc.get(key)
        if not isinstance(nested_doc, dict):
            continue
        # training_config artifacts are often {available, path, data}
        if key == "training_config" and isinstance(nested_doc.get("data"), dict):
            nested_doc = nested_doc["data"]
        if id(nested_doc) in seen:
            continue
        nested = extract_feature_selection_lineage(
            nested_doc, _depth=_depth + 1, _seen=seen
        )
        if nested:
            return nested
    return None


def _load_discovery_scores(data_dir: str, run_id: str) -> dict[str, float]:
    with _AnalysisDb(data_dir) as conn:
        rows = conn.execute(
            """
            SELECT feature_name, feature_score, rating_score
            FROM feature_profiles
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
    out: dict[str, float] = {}
    for r in rows:
        name = str(r["feature_name"] or "")
        sc = r["feature_score"]
        if sc is None:
            sc = r["rating_score"]
        if name and sc is not None:
            out[name] = float(sc)
    return out


def _load_all_feature_names(data_dir: str, run_id: str) -> list[str]:
    with _AnalysisDb(data_dir) as conn:
        rows = conn.execute(
            """
            SELECT feature_name FROM feature_profiles
            WHERE run_id = ?
            ORDER BY feature_name
            """,
            (run_id,),
        ).fetchall()
    names = [str(r["feature_name"]) for r in rows if r["feature_name"]]
    if names:
        return names
    with _AnalysisDb(data_dir) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT feature FROM (
                SELECT feature_a AS feature FROM correlation WHERE run_id = ?
                UNION
                SELECT feature_b AS feature FROM correlation WHERE run_id = ?
            )
            ORDER BY feature
            """,
            (run_id, run_id),
        ).fetchall()
    return [str(r["feature"]) for r in rows if r["feature"]]


def _load_perm_importance(data_dir: str, run_id: str) -> dict[str, float]:
    with _AnalysisDb(data_dir) as conn:
        rows = conn.execute(
            """
            SELECT feature_name, importance, delta_rmse
            FROM permutation_importance
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
    out: dict[str, float] = {}
    for r in rows:
        name = str(r["feature_name"] or "")
        if not name:
            continue
        imp = r["importance"]
        if imp is None:
            imp = r["delta_rmse"]
        if imp is None:
            continue
        val = float(imp)
        prev = out.get(name)
        if prev is None or abs(val) > abs(prev):
            out[name] = val
    return out


def _load_corr_pairs(
    data_dir: str, run_id: str, *, min_abs: float
) -> list[tuple[str, str, float]]:
    with _AnalysisDb(data_dir) as conn:
        rows = conn.execute(
            """
            SELECT feature_a, feature_b, correlation
            FROM correlation
            WHERE run_id = ? AND ABS(correlation) >= ?
            ORDER BY ABS(correlation) DESC
            """,
            (run_id, float(min_abs)),
        ).fetchall()
    out: list[tuple[str, str, float]] = []
    for r in rows:
        a, b = str(r["feature_a"]), str(r["feature_b"])
        if a and b and a != b:
            out.append((a, b, abs(float(r["correlation"] or 0.0))))
    return out


def _score_map_for_keep(
    discovery: dict[str, float],
    perm: dict[str, float],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in set(discovery) | set(perm):
        d = discovery.get(name)
        p = perm.get(name)
        if d is not None:
            out[name] = float(d)
        elif p is not None:
            out[name] = abs(float(p)) * 100.0
        else:
            out[name] = 0.0
    return out


def correlation_filter(
    features: list[str],
    pairs: list[tuple[str, str, float]],
    scores: dict[str, float],
    *,
    threshold: float,
) -> list[str]:
    """Greedy drop: for each high-|r| pair, drop the lower-scoring feature."""
    remaining = set(features)
    ordered = sorted(pairs, key=lambda t: -t[2])
    for a, b, abs_r in ordered:
        if abs_r < float(threshold):
            continue
        if a not in remaining or b not in remaining:
            continue
        sa = float(scores.get(a, 0.0))
        sb = float(scores.get(b, 0.0))
        drop = b if sa >= sb else a
        remaining.discard(drop)
    return [f for f in features if f in remaining]


def permutation_filter(
    features: list[str],
    perm: dict[str, float],
    *,
    threshold: float,
) -> list[str]:
    """Keep features with |importance| >= threshold."""
    thr = float(threshold)
    kept: list[str] = []
    for f in features:
        if f not in perm:
            continue
        if abs(float(perm[f])) >= thr:
            kept.append(f)
    return kept


def _select_hca_top_n(
    data_dir: str,
    run_id: str,
    *,
    top_n: int,
    min_family_size: int,
) -> tuple[list[str], list[dict[str, Any]], dict[str, str]]:
    from .analysis_family_review import (
        FILTER_ALL,
        current_experiment_reps,
        load_families_with_reviews,
        rank_family_members_by_score,
    )

    scores = _load_discovery_scores(data_dir, run_id)
    families = load_families_with_reviews(
        data_dir, run_id, min_size=min_family_size, status_filter=FILTER_ALL
    )
    primary = current_experiment_reps(
        data_dir, run_id, min_size=min_family_size
    )
    selected: list[str] = []
    family_rows: list[dict[str, Any]] = []
    n = max(int(top_n), 1)
    for fam in families:
        fid = str(fam.get("family_id") or "")
        members = [str(m) for m in (fam.get("members") or []) if m]
        ranked = rank_family_members_by_score(members, scores)
        primary_rep = primary.get(fid) or str(
            fam.get("experiment_representative")
            or fam.get("suggested_representative")
            or ""
        ).strip()
        picks: list[str] = []
        if primary_rep and primary_rep in members:
            picks.append(primary_rep)
        for name, _sc in ranked:
            if name in picks:
                continue
            picks.append(name)
            if len(picks) >= n:
                break
        picks = picks[:n]
        for name in picks:
            if name not in selected:
                selected.append(name)
        family_rows.append(
            {
                "family_id": fid,
                "family_label": fam.get("family_label") or fid,
                "members": members,
                "representative": picks[0] if picks else primary_rep or None,
                "representatives": picks,
                "n_selected": len(picks),
            }
        )
    family_reps = {
        str(r["family_id"]): str(r["representative"])
        for r in family_rows
        if r.get("family_id") and r.get("representative")
    }
    return selected, family_rows, family_reps


def build_final_feature_dataset(
    data_dir: str,
    run_id: str,
    config: dict[str, Any] | None = None,
    *,
    strategy: str | None = None,
    representative_policy: str | None = None,
    top_n: int | None = None,
    correlation_threshold: float | None = None,
    permutation_threshold: float | None = None,
    min_family_size: int = 2,
) -> dict[str, Any]:
    """Resolve a Final Feature Dataset for the given strategy config."""
    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")

    base = dict(config or {})
    cfg = build_selection_config(
        strategy or base.get("strategy") or STRATEGY_HCA,
        representative_policy=str(
            representative_policy
            or base.get("representative_policy")
            or POLICY_TOP_1
        ),
        top_n=top_n if top_n is not None else base.get("top_n"),
        correlation_threshold=(
            float(correlation_threshold)
            if correlation_threshold is not None
            else float(base.get("correlation_threshold", DEFAULT_CORR_THRESHOLD))
        ),
        permutation_threshold=(
            float(permutation_threshold)
            if permutation_threshold is not None
            else float(base.get("permutation_threshold", DEFAULT_PERM_THRESHOLD))
        ),
        min_family_size=int(
            base.get("min_family_size")
            if base.get("min_family_size") is not None
            else min_family_size
        ),
    )
    sid = cfg["strategy"]
    all_feats = _load_all_feature_names(data_dir, rid)
    discovery = _load_discovery_scores(data_dir, rid)
    perm = _load_perm_importance(data_dir, rid)
    scores = _score_map_for_keep(discovery, perm)
    corr_thr = float(cfg["correlation_threshold"])
    perm_thr = float(cfg["permutation_threshold"])

    family_rows: list[dict[str, Any]] = []
    family_reps: dict[str, str] = {}
    selected: list[str] = []
    pipeline: list[str] = []

    if sid == STRATEGY_HCA:
        pipeline = [
            f"{len(all_feats)} Features",
            "Correlation",
            "HCA Families",
            f"Representative Policy ({cfg.get('representative_policy_label')})",
            "Final Feature Dataset",
        ]
        pairs = _load_corr_pairs(data_dir, rid, min_abs=corr_thr)
        after_corr = correlation_filter(
            all_feats, pairs, scores, threshold=corr_thr
        )
        cfg["n_after_correlation"] = len(after_corr)
        selected, family_rows, family_reps = _select_hca_top_n(
            data_dir,
            rid,
            top_n=int(cfg.get("top_n") or 1),
            min_family_size=int(cfg.get("min_family_size") or 2),
        )
        cfg["n_families"] = len(family_rows)
    elif sid == STRATEGY_CORR_PERM:
        pipeline = [
            f"{len(all_feats)} Features",
            "Correlation Filter",
            "Permutation Ranking",
            "Keep Features",
            "Final Feature Dataset",
        ]
        pairs = _load_corr_pairs(data_dir, rid, min_abs=corr_thr)
        after_corr = correlation_filter(
            all_feats, pairs, scores, threshold=corr_thr
        )
        selected = permutation_filter(after_corr, perm, threshold=perm_thr)
        cfg["n_after_correlation"] = len(after_corr)
        cfg["n_after_permutation"] = len(selected)
    elif sid == STRATEGY_PERM_ONLY:
        pipeline = [
            f"{len(all_feats)} Features",
            "Permutation Ranking",
            "Keep Features",
            "Final Feature Dataset",
        ]
        ranked = sorted(
            ((f, abs(float(perm[f]))) for f in all_feats if f in perm),
            key=lambda t: (-t[1], t[0]),
        )
        selected = [f for f, v in ranked if v >= perm_thr]
        cfg["n_after_permutation"] = len(selected)
    elif sid == STRATEGY_CORR_ONLY:
        pipeline = [
            f"{len(all_feats)} Features",
            "Correlation Filter",
            "Final Feature Dataset",
        ]
        pairs = _load_corr_pairs(data_dir, rid, min_abs=corr_thr)
        selected = correlation_filter(
            all_feats, pairs, scores, threshold=corr_thr
        )
        cfg["n_after_correlation"] = len(selected)
    else:
        raise ValueError(f"Unknown feature selection strategy: {sid!r}")

    selected = list(dict.fromkeys(str(f) for f in selected if f))
    cfg["n_selected_features"] = len(selected)
    cfg["n_input_features"] = len(all_feats)
    cfg["resolved_at"] = _now_iso()

    from .analysis_experiments import build_feature_set, features_fingerprint

    feature_set = build_feature_set(
        families=family_rows if family_rows else None,
        features=selected,
    )
    if not family_rows:
        feature_set["families"] = []
        feature_set["features"] = selected
        feature_set["count"] = len(selected)
        feature_set["hash"] = features_fingerprint(selected) if selected else ""

    return {
        "kind": "final_feature_dataset",
        "count": len(selected),
        "features": selected,
        "hash": feature_set.get("hash")
        or (features_fingerprint(selected) if selected else ""),
        "feature_selection": cfg,
        "feature_set": feature_set,
        "family_reps": family_reps,
        "families": family_rows,
        "pipeline": pipeline,
        "n_input_features": len(all_feats),
    }


def preview_selection(
    data_dir: str,
    run_id: str,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Lightweight preview for UI (counts + config + feature list)."""
    ds = build_final_feature_dataset(data_dir, run_id, config, **kwargs)
    cfg = dict(ds.get("feature_selection") or {})
    return {
        "strategy": cfg.get("strategy"),
        "strategy_label": cfg.get("strategy_label"),
        "representative_policy": cfg.get("representative_policy"),
        "representative_policy_label": cfg.get("representative_policy_label"),
        "top_n": cfg.get("top_n"),
        "correlation_threshold": cfg.get("correlation_threshold"),
        "permutation_threshold": cfg.get("permutation_threshold"),
        "n_input_features": ds.get("n_input_features"),
        "n_features": ds.get("count"),
        "n_families": cfg.get("n_families"),
        "n_after_correlation": cfg.get("n_after_correlation"),
        "n_after_permutation": cfg.get("n_after_permutation"),
        "pipeline": ds.get("pipeline"),
        "features": list(ds.get("features") or []),
        "hash": ds.get("hash"),
        "summary_text": format_selection_summary(
            cfg, n_features=int(ds.get("count") or 0)
        ),
        "dataset": ds,
    }


def compare_strategy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize rows for the Strategy comparison table."""
    out: list[dict[str, Any]] = []
    for r in rows:
        cfg = dict(r.get("feature_selection") or {})
        sid = normalize_strategy(cfg.get("strategy") or r.get("strategy"))
        label = cfg.get("strategy_short") or STRATEGY_SHORT.get(sid, sid)
        if sid == STRATEGY_HCA:
            pol = cfg.get("representative_policy_label") or POLICY_LABELS.get(
                str(cfg.get("representative_policy") or ""), ""
            )
            if pol:
                label = f"HCA {pol}"
        out.append(
            {
                "strategy": sid,
                "strategy_label": label,
                "n_features": r.get("n_features")
                or r.get("count")
                or cfg.get("n_selected_features"),
                "holdout": r.get("holdout")
                or r.get("holdout_score")
                or r.get("holdout_r2"),
                "walk_forward": r.get("walk_forward")
                or r.get("walk_forward_score")
                or r.get("walk_forward_r2"),
                "validation": r.get("validation") or r.get("validation_label"),
                "trading_result": r.get("trading_result"),
                "experiment_id": r.get("experiment_id"),
                "champion": r.get("champion") or r.get("is_champion"),
            }
        )
    return out


def parse_selection_config_json(raw: Any) -> dict[str, Any] | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, dict):
        kwargs: dict[str, Any] = {}
        for k in (
            "strategy",
            "representative_policy",
            "top_n",
            "correlation_threshold",
            "permutation_threshold",
            "min_family_size",
        ):
            if k in raw:
                kwargs[k] = raw[k]
        return build_selection_config(**kwargs)
    try:
        data = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        return parse_selection_config_json(data)
    return None


__all__ = [
    "ALL_STRATEGIES",
    "DEFAULT_CORR_THRESHOLD",
    "DEFAULT_PERM_THRESHOLD",
    "DEFAULT_TOP_N",
    "POLICY_LABELS",
    "POLICY_TOP_1",
    "POLICY_TOP_2",
    "POLICY_TOP_3",
    "POLICY_TOP_N",
    "STRATEGY_CORR_ONLY",
    "STRATEGY_CORR_PERM",
    "STRATEGY_HCA",
    "STRATEGY_LABELS",
    "STRATEGY_PERM_ONLY",
    "STRATEGY_SHORT",
    "build_final_feature_dataset",
    "build_feature_selection_lineage",
    "build_selection_config",
    "compare_strategy_rows",
    "correlation_filter",
    "extract_feature_selection_lineage",
    "feature_selection_overview_rows",
    "format_selection_summary",
    "normalize_policy",
    "normalize_strategy",
    "parse_selection_config_json",
    "permutation_filter",
    "preview_selection",
    "required_modules_for_strategy",
]
