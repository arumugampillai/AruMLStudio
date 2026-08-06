"""Correlation Insights — actionable recommendations from correlation clusters.

Read-only diagnostics. Never deletes or modifies features/datasets/models.
Stage 1 only — next steps are MI / Permutation / Discovery Rating.
Stage 2 SHAP may attach enrichment notes without overwriting correlation reasons.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from .analysis_lab_store import _AnalysisDb, _now_iso

# Recommendation labels (never "Delete")
REC_KEEP = "Keep"
REC_REVIEW = "Review"
REC_DUPLICATE = "Duplicate Candidate"
REC_LARGE_FAMILY = "Large feature family"

DUPLICATE_CORR = 0.9999
LARGE_CLUSTER_N = 10


def ensure_correlation_insights_schema(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS correlation_insights (
            run_id TEXT NOT NULL,
            cluster TEXT NOT NULL,
            family TEXT NOT NULL DEFAULT '',
            members INTEGER NOT NULL DEFAULT 0,
            members_json TEXT NOT NULL DEFAULT '[]',
            max_correlation REAL,
            avg_correlation REAL,
            representative TEXT,
            recommendation TEXT NOT NULL DEFAULT 'Keep',
            reason TEXT NOT NULL DEFAULT '',
            flags_json TEXT NOT NULL DEFAULT '{}',
            duplicate_pair_json TEXT,
            shap_enrichment TEXT,
            vif_enrichment TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, cluster)
        )
        """
    )


def _pair_stats(
    members: Sequence[str],
    pairs: Sequence[tuple[str, str, float]],
) -> tuple[float, float, list[tuple[str, str, float]]]:
    """Return (max_|r|, avg_|r|, intra-cluster pairs)."""
    member_set = set(members)
    intra: list[tuple[str, str, float]] = []
    for a, b, r in pairs:
        if a in member_set and b in member_set:
            intra.append((a, b, float(r)))
    if not intra:
        return (1.0 if len(members) <= 1 else 0.0), (
            1.0 if len(members) <= 1 else 0.0
        ), []
    abs_vals = [abs(r) for _, _, r in intra]
    return max(abs_vals), (sum(abs_vals) / len(abs_vals)), intra


def _dominant_family(members: Sequence[str], family_of) -> str:  # type: ignore[no-untyped-def]
    votes: dict[str, int] = {}
    for m in members:
        fam = str(family_of(m) or "Other")
        votes[fam] = votes.get(fam, 0) + 1
    return max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _same_family(members: Sequence[str], family_of) -> bool:  # type: ignore[no-untyped-def]
    fams = {str(family_of(m) or "Other") for m in members}
    return len(fams) == 1


def recommend_for_cluster(
    *,
    members: Sequence[str],
    max_corr: float,
    avg_corr: float,
    family: str,
    same_family: bool,
    intra_pairs: Sequence[tuple[str, str, float]],
) -> dict[str, Any]:
    """Apply Correlation Insights rules (investigation only — never Delete)."""
    n = len(members)
    flags: dict[str, bool] = {
        "possible_mathematical_duplicates": False,
        "large_feature_family": False,
        "review_after_discovery": False,
        "investigate_duplicate_implementation": False,
    }
    duplicate_pair: tuple[str, str, float] | None = None

    # Two-feature near-perfect same-family → duplicate candidate
    if n == 2 and max_corr >= DUPLICATE_CORR and same_family:
        flags["possible_mathematical_duplicates"] = True
        flags["investigate_duplicate_implementation"] = True
        # Prefer the pair with highest |r|
        if intra_pairs:
            a, b, r = max(intra_pairs, key=lambda t: abs(t[2]))
            duplicate_pair = (a, b, float(r))
        else:
            duplicate_pair = (str(members[0]), str(members[1]), float(max_corr))
        reason = (
            f"{duplicate_pair[0]} and {duplicate_pair[1]} appear mathematically "
            f"equivalent (|r|={abs(duplicate_pair[2]):.4f}). "
            "Possible mathematical duplicates — review equations. "
            "Investigate duplicate implementation. "
            "Do not remove features based on correlation alone."
        )
        return {
            "recommendation": REC_DUPLICATE,
            "reason": reason,
            "flags": flags,
            "duplicate_pair": duplicate_pair,
        }

    # Large cluster → continue through Discovery (MI / Permutation / Rating)
    if n > LARGE_CLUSTER_N:
        flags["large_feature_family"] = True
        flags["review_after_discovery"] = True
        reason = (
            f"Large cluster containing {n} {family.lower()}-derived features "
            f"(max |r|={max_corr:.4f}, avg |r|={avg_corr:.4f}). "
            "Continue with HCA → MI → Permutation → Discovery Rating before "
            "removing any features. "
            "Correlation is an investigation tool, not a feature selection tool."
        )
        # Prefer explicit "Review" as in the product example; tag large family in flags.
        return {
            "recommendation": REC_REVIEW,
            "reason": reason,
            "flags": flags,
            "duplicate_pair": None,
            "recommendation_tag": REC_LARGE_FAMILY,
        }

    # Near-perfect pair but mixed families — still worth a look
    if n == 2 and max_corr >= DUPLICATE_CORR and not same_family:
        flags["investigate_duplicate_implementation"] = True
        a, b = str(members[0]), str(members[1])
        reason = (
            f"{a} and {b} are almost perfectly correlated (|r|={max_corr:.4f}) "
            "but span different feature families. Investigate duplicate "
            "implementation or shared driver. Do not delete from correlation alone."
        )
        return {
            "recommendation": REC_REVIEW,
            "reason": reason,
            "flags": flags,
            "duplicate_pair": (a, b, float(max_corr)),
        }

    # Default: keep — correlation alone is not selection
    reason = (
        f"{n} feature(s); max |r|={max_corr:.4f}, avg |r|={avg_corr:.4f}. "
        "No correlation red flag. Keep for now — wait for MI / Permutation / "
        "Discovery Rating before any removal decisions."
    )
    return {
        "recommendation": REC_KEEP,
        "reason": reason,
        "flags": flags,
        "duplicate_pair": None,
    }


def build_correlation_insights(
    *,
    clusters: Sequence[dict[str, Any]],
    pairs: Sequence[tuple[str, str, float]],
    family_of=None,  # type: ignore[no-untyped-def]
) -> list[dict[str, Any]]:
    """Build insight records for multi-member clusters (and optionally singletons)."""
    if family_of is None:
        from .analysis_correlation import _family_label as family_of

    out: list[dict[str, Any]] = []
    for raw in clusters:
        members = [str(m) for m in (raw.get("members") or []) if str(m).strip()]
        if len(members) < 2:
            # Singletons: Keep, low noise in UI — skip or include lightly
            continue
        cluster = str(raw.get("cluster") or "")
        family = _dominant_family(members, family_of)
        max_corr, avg_corr, intra = _pair_stats(members, pairs)
        # Prefer precomputed highest if present and higher fidelity
        if raw.get("highest_correlation") is not None:
            try:
                max_corr = max(max_corr, float(raw["highest_correlation"]))
            except (TypeError, ValueError):
                pass
        rep = str(raw.get("representative") or "") or (members[0] if members else "")
        decision = recommend_for_cluster(
            members=members,
            max_corr=max_corr,
            avg_corr=avg_corr,
            family=family,
            same_family=_same_family(members, family_of),
            intra_pairs=intra,
        )
        out.append(
            {
                "cluster": cluster,
                "family": family,
                "members": members,
                "member_count": len(members),
                "max_correlation": float(max_corr),
                "avg_correlation": float(avg_corr),
                "representative": rep,
                "recommendation": decision["recommendation"],
                "recommendation_tag": decision.get("recommendation_tag") or "",
                "reason": decision["reason"],
                "flags": decision["flags"],
                "duplicate_pair": decision.get("duplicate_pair"),
            }
        )
    # Sort: Duplicate / Review first, then by size
    rank = {REC_DUPLICATE: 0, REC_REVIEW: 1, REC_LARGE_FAMILY: 1, REC_KEEP: 2}

    def _key(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            rank.get(str(item["recommendation"]), 9),
            -int(item["member_count"]),
            -float(item["max_correlation"]),
            str(item["cluster"]),
        )

    out.sort(key=_key)
    return out


def persist_correlation_insights(
    data_dir: str,
    run_id: str,
    insights: Sequence[dict[str, Any]],
) -> int:
    """Replace insight rows for ``run_id``. Returns row count written."""
    now = _now_iso()
    with _AnalysisDb(data_dir) as conn:
        ensure_correlation_insights_schema(conn)
        conn.execute(
            "DELETE FROM correlation_insights WHERE run_id = ?",
            (run_id,),
        )
        rows = []
        for item in insights:
            dup = item.get("duplicate_pair")
            rows.append(
                (
                    run_id,
                    str(item.get("cluster") or ""),
                    str(item.get("family") or ""),
                    int(item.get("member_count") or len(item.get("members") or [])),
                    json.dumps(list(item.get("members") or []), separators=(",", ":")),
                    float(item.get("max_correlation") or 0.0),
                    float(item.get("avg_correlation") or 0.0),
                    str(item.get("representative") or ""),
                    str(item.get("recommendation") or REC_KEEP),
                    str(item.get("reason") or ""),
                    json.dumps(item.get("flags") or {}, separators=(",", ":")),
                    json.dumps(list(dup), separators=(",", ":")) if dup else None,
                    None,  # shap_enrichment
                    None,  # vif_enrichment
                    now,
                )
            )
        conn.executemany(
            """
            INSERT INTO correlation_insights (
                run_id, cluster, family, members, members_json,
                max_correlation, avg_correlation, representative,
                recommendation, reason, flags_json, duplicate_pair_json,
                shap_enrichment, vif_enrichment, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def load_correlation_insights(
    data_dir: str,
    run_id: str,
) -> list[dict[str, Any]]:
    with _AnalysisDb(data_dir) as conn:
        ensure_correlation_insights_schema(conn)
        rows = conn.execute(
            """
            SELECT * FROM correlation_insights
            WHERE run_id = ?
            ORDER BY
                CASE recommendation
                    WHEN 'Duplicate Candidate' THEN 0
                    WHEN 'Review' THEN 1
                    WHEN 'Large feature family' THEN 1
                    ELSE 2
                END,
                members DESC,
                max_correlation DESC,
                cluster ASC
            """,
            (run_id,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        item = dict(r)
        try:
            item["members_list"] = json.loads(str(item.get("members_json") or "[]"))
        except Exception:
            item["members_list"] = []
        try:
            item["flags"] = json.loads(str(item.get("flags_json") or "{}"))
        except Exception:
            item["flags"] = {}
        out.append(item)
    return out


def enrich_insight(
    data_dir: str,
    run_id: str,
    cluster: str,
    *,
    shap_enrichment: str | None = None,
    vif_enrichment: str | None = None,
) -> None:
    """Later SHAP/VIF modules attach notes without overwriting correlation reason."""
    with _AnalysisDb(data_dir) as conn:
        ensure_correlation_insights_schema(conn)
        if shap_enrichment is not None:
            conn.execute(
                """
                UPDATE correlation_insights
                SET shap_enrichment = ?
                WHERE run_id = ? AND cluster = ?
                """,
                (shap_enrichment, run_id, cluster),
            )
        if vif_enrichment is not None:
            conn.execute(
                """
                UPDATE correlation_insights
                SET vif_enrichment = ?
                WHERE run_id = ? AND cluster = ?
                """,
                (vif_enrichment, run_id, cluster),
            )


def rebuild_insights_from_stored_correlation(
    data_dir: str,
    run_id: str,
) -> list[dict[str, Any]]:
    """Build Insights from existing correlation/clusters rows (no parquet recompute).

    Useful when Correlation completed before Insights existed.
    """
    from .analysis_correlation import load_clusters
    from .analysis_lab_store import _AnalysisDb

    clusters = load_clusters(data_dir, run_id)
    if not clusters:
        return []
    with _AnalysisDb(data_dir) as conn:
        rows = conn.execute(
            """
            SELECT feature_a, feature_b, correlation
            FROM correlation
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
    pairs = [
        (str(r["feature_a"]), str(r["feature_b"]), float(r["correlation"]))
        for r in rows
    ]
    # Attach size for builders that expect it
    for c in clusters:
        c["size"] = int(c.get("size") or len(c.get("members") or []))
        if c.get("highest_correlation") is None:
            c["highest_correlation"] = 0.0
    insights = build_correlation_insights(clusters=clusters, pairs=pairs)
    persist_correlation_insights(data_dir, run_id, insights)
    return insights


__all__ = [
    "DUPLICATE_CORR",
    "LARGE_CLUSTER_N",
    "REC_DUPLICATE",
    "REC_KEEP",
    "REC_LARGE_FAMILY",
    "REC_REVIEW",
    "build_correlation_insights",
    "enrich_insight",
    "ensure_correlation_insights_schema",
    "load_correlation_insights",
    "persist_correlation_insights",
    "rebuild_insights_from_stored_correlation",
    "recommend_for_cluster",
]
