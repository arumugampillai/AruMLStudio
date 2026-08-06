"""Hierarchical Cluster Analysis (HCA) for Feature Discovery.

Layer responsibility
--------------------
Question: Which features form families?
Output:   Feature families + representative *candidates* (ranked evidence only)
Must not: Finalize the feature set or lock a representative

Builds agglomerative clusters from stored Correlation pairs using distance
``d = 1 - |r|``. Does not recompute correlation and does not auto-select the
final representative for training.
"""
from __future__ import annotations

import json
import math
import time
from typing import Any, Callable, Sequence

from .analysis_correlation import _family_label, load_top_pairs
from .analysis_lab_store import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    _AnalysisDb,
    _now_iso,
    set_module_status,
)

ProgressCb = Callable[[dict[str, Any]], None]

# Cut families when average-linkage distance exceeds this (≈ |r| < 0.85).
DEFAULT_DISTANCE_THRESHOLD = 0.15
DEFAULT_LINKAGE = "average"
DEFAULT_CANDIDATE_LIMIT = 8


def ensure_hca_schema(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_families (
            run_id TEXT NOT NULL,
            family_id TEXT NOT NULL,
            family_label TEXT,
            size INTEGER NOT NULL DEFAULT 0,
            max_corr REAL,
            avg_corr REAL,
            members_json TEXT,
            candidate_reps_json TEXT,
            params_json TEXT,
            created_at TEXT,
            PRIMARY KEY (run_id, family_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_family_members (
            run_id TEXT NOT NULL,
            family_id TEXT NOT NULL,
            feature TEXT NOT NULL,
            candidate_rank INTEGER,
            avg_corr_to_family REAL,
            PRIMARY KEY (run_id, feature)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_family_members_family
        ON feature_family_members(run_id, family_id)
        """
    )


def _load_all_pairs(data_dir: str, run_id: str) -> list[tuple[str, str, float]]:
    rows = load_top_pairs(data_dir, run_id, limit=2_000_000, min_abs=0.0)
    out: list[tuple[str, str, float]] = []
    for r in rows:
        a = str(r.get("feature_a") or "")
        b = str(r.get("feature_b") or "")
        try:
            corr = float(r.get("correlation"))
        except (TypeError, ValueError):
            continue
        if not a or not b or math.isnan(corr):
            continue
        out.append((a, b, corr))
    return out


def _features_from_pairs(pairs: Sequence[tuple[str, str, float]]) -> list[str]:
    names: set[str] = set()
    for a, b, _ in pairs:
        names.add(a)
        names.add(b)
    return sorted(names)


def _corr_lookup(
    pairs: Sequence[tuple[str, str, float]],
) -> dict[tuple[str, str], float]:
    lk: dict[tuple[str, str], float] = {}
    for a, b, r in pairs:
        key = (a, b) if a < b else (b, a)
        lk[key] = float(r)
    return lk


def _abs_corr(
    lookup: dict[tuple[str, str], float], a: str, b: str
) -> float | None:
    if a == b:
        return 1.0
    key = (a, b) if a < b else (b, a)
    if key not in lookup:
        return None
    return abs(float(lookup[key]))


def _candidate_scores(
    members: Sequence[str],
    lookup: dict[tuple[str, str], float],
) -> list[dict[str, Any]]:
    """Rank members by mean |r| to other family members (evidence only)."""
    scored: list[dict[str, Any]] = []
    mem = [str(m) for m in members]
    for feat in mem:
        vals: list[float] = []
        for other in mem:
            if other == feat:
                continue
            c = _abs_corr(lookup, feat, other)
            if c is not None:
                vals.append(c)
        avg = sum(vals) / len(vals) if vals else 0.0
        scored.append(
            {
                "feature": feat,
                "avg_corr_to_family": float(avg),
                "n_links": len(vals),
            }
        )
    scored.sort(
        key=lambda x: (-float(x["avg_corr_to_family"]), str(x["feature"]))
    )
    for i, row in enumerate(scored, start=1):
        row["candidate_rank"] = i
    return scored


def _family_stats(
    members: Sequence[str],
    lookup: dict[tuple[str, str], float],
) -> tuple[float | None, float | None]:
    vals: list[float] = []
    mem = list(members)
    for i, a in enumerate(mem):
        for b in mem[i + 1 :]:
            c = _abs_corr(lookup, a, b)
            if c is not None:
                vals.append(c)
    if not vals:
        return None, None
    return max(vals), sum(vals) / len(vals)


def _vote_family_label(members: Sequence[str]) -> str:
    counts: dict[str, int] = {}
    for m in members:
        lab = _family_label(m)
        counts[lab] = counts.get(lab, 0) + 1
    if not counts:
        return "Other"
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def compute_hca_families(
    pairs: Sequence[tuple[str, str, float]],
    *,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    linkage_method: str = DEFAULT_LINKAGE,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> list[dict[str, Any]]:
    """Run agglomerative clustering; return family dicts (no final representative)."""
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    features = _features_from_pairs(pairs)
    if len(features) < 2:
        return []
    lookup = _corr_lookup(pairs)
    n = len(features)
    idx = {f: i for i, f in enumerate(features)}
    dist = np.ones((n, n), dtype=float)
    np.fill_diagonal(dist, 0.0)
    for a, b, r in pairs:
        i, j = idx[a], idx[b]
        d = 1.0 - abs(float(r))
        if d < 0.0:
            d = 0.0
        if d > 1.0:
            d = 1.0
        dist[i, j] = d
        dist[j, i] = d

    condensed = squareform(dist, checks=False)
    # Average linkage keeps distance interpretation close to 1-|r|.
    Z = linkage(condensed, method=str(linkage_method or DEFAULT_LINKAGE))
    labels = fcluster(
        Z,
        t=float(distance_threshold),
        criterion="distance",
    )

    by_lab: dict[int, list[str]] = {}
    for feat, lab in zip(features, labels):
        by_lab.setdefault(int(lab), []).append(str(feat))

    # Stable order: larger families first, then label vote, then id
    raw_groups = sorted(
        by_lab.values(),
        key=lambda mem: (-len(mem), _vote_family_label(mem), sorted(mem)[0]),
    )

    families: list[dict[str, Any]] = []
    label_counts: dict[str, int] = {}
    for i, members in enumerate(raw_groups, start=1):
        members = sorted(members)
        base = _vote_family_label(members)
        label_counts[base] = label_counts.get(base, 0) + 1
        suffix = label_counts[base]
        pretty = f"{base} Family" if suffix == 1 else f"{base} Family {suffix}"
        family_id = f"HCA-{i:03d}"
        max_c, avg_c = _family_stats(members, lookup)
        candidates = _candidate_scores(members, lookup)[: int(candidate_limit)]
        families.append(
            {
                "family_id": family_id,
                "family_label": pretty,
                "size": len(members),
                "max_corr": max_c,
                "avg_corr": avg_c,
                "members": members,
                "candidates": candidates,
            }
        )
    return families


def persist_hca_results(
    data_dir: str,
    run_id: str,
    families: Sequence[dict[str, Any]],
    *,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stamp = _now_iso()
    params_json = json.dumps(params or {}, separators=(",", ":"))
    with _AnalysisDb(data_dir) as conn:
        ensure_hca_schema(conn)
        conn.execute(
            "DELETE FROM feature_family_members WHERE run_id = ?", (run_id,)
        )
        conn.execute(
            "DELETE FROM feature_families WHERE run_id = ?", (run_id,)
        )
        for fam in families:
            fid = str(fam["family_id"])
            members = list(fam.get("members") or [])
            candidates = list(fam.get("candidates") or [])
            conn.execute(
                """
                INSERT INTO feature_families (
                    run_id, family_id, family_label, size, max_corr, avg_corr,
                    members_json, candidate_reps_json, params_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    fid,
                    str(fam.get("family_label") or ""),
                    int(fam.get("size") or len(members)),
                    fam.get("max_corr"),
                    fam.get("avg_corr"),
                    json.dumps(members, separators=(",", ":")),
                    json.dumps(candidates, separators=(",", ":")),
                    params_json,
                    stamp,
                ),
            )
            cand_by_feat = {
                str(c.get("feature")): c for c in candidates if c.get("feature")
            }
            for feat in members:
                c = cand_by_feat.get(feat) or {}
                conn.execute(
                    """
                    INSERT INTO feature_family_members (
                        run_id, family_id, feature, candidate_rank,
                        avg_corr_to_family
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        fid,
                        feat,
                        c.get("candidate_rank"),
                        c.get("avg_corr_to_family"),
                    ),
                )
    multi = sum(1 for f in families if int(f.get("size") or 0) >= 2)
    return {
        "n_families": len(families),
        "n_multi_member": multi,
        "n_singletons": len(families) - multi,
    }


def load_families(
    data_dir: str,
    run_id: str,
    *,
    min_size: int = 1,
) -> list[dict[str, Any]]:
    with _AnalysisDb(data_dir) as conn:
        ensure_hca_schema(conn)
        rows = conn.execute(
            """
            SELECT * FROM feature_families
            WHERE run_id = ? AND size >= ?
            ORDER BY size DESC, family_id ASC
            """,
            (run_id, int(min_size)),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        item = dict(r)
        try:
            item["members"] = json.loads(str(item.get("members_json") or "[]"))
        except Exception:
            item["members"] = []
        try:
            item["candidates"] = json.loads(
                str(item.get("candidate_reps_json") or "[]")
            )
        except Exception:
            item["candidates"] = []
        out.append(item)
    return out


def load_family_members(
    data_dir: str, run_id: str, family_id: str
) -> list[dict[str, Any]]:
    with _AnalysisDb(data_dir) as conn:
        ensure_hca_schema(conn)
        rows = conn.execute(
            """
            SELECT * FROM feature_family_members
            WHERE run_id = ? AND family_id = ?
            ORDER BY
                CASE WHEN candidate_rank IS NULL THEN 1 ELSE 0 END,
                candidate_rank ASC,
                feature ASC
            """,
            (run_id, family_id),
        ).fetchall()
        return [dict(r) for r in rows]


def families_exist(data_dir: str, run_id: str) -> bool:
    with _AnalysisDb(data_dir) as conn:
        ensure_hca_schema(conn)
        row = conn.execute(
            "SELECT 1 FROM feature_families WHERE run_id = ? LIMIT 1",
            (run_id,),
        ).fetchone()
        return row is not None


def run_hca_analysis(
    data_dir: str,
    run_id: str,
    *,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    linkage_method: str = DEFAULT_LINKAGE,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Build Feature Families from stored Correlation pairs."""
    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")

    started = _now_iso()
    t0 = time.perf_counter()

    def _tick(frac: float, message: str, **extra: Any) -> None:
        if not progress:
            return
        progress(
            {
                "frac": max(0.0, min(1.0, float(frac))),
                "elapsed": max(time.perf_counter() - t0, 0.0),
                "message": str(message),
                **extra,
            }
        )

    set_module_status(
        data_dir,
        rid,
        "hca",
        STATUS_RUNNING,
        started_at=started,
        message="Building Feature Families from Correlation…",
    )
    _tick(0.02, "Starting HCA…")
    try:
        _tick(0.10, "Loading correlation pairs…")
        pairs = _load_all_pairs(data_dir, rid)
        if not pairs:
            raise ValueError(
                "No correlation pairs found — run Correlation first."
            )
        params = {
            "distance_threshold": float(distance_threshold),
            "approx_min_abs_corr": 1.0 - float(distance_threshold),
            "linkage": str(linkage_method),
            "candidate_limit": int(candidate_limit),
            "n_pairs": len(pairs),
        }
        _tick(0.35, f"Clustering {len(pairs):,} pairs…")
        families = compute_hca_families(
            pairs,
            distance_threshold=float(distance_threshold),
            linkage_method=str(linkage_method),
            candidate_limit=int(candidate_limit),
        )
        _tick(0.85, f"Persisting {len(families)} families…")
        summary = persist_hca_results(data_dir, rid, families, params=params)
        elapsed = round(max(time.perf_counter() - t0, 0.0), 3)
        msg = (
            f"HCA · {summary['n_families']} families "
            f"({summary['n_multi_member']} multi-member, "
            f"{summary['n_singletons']} singleton) · "
            f"candidates only — no auto-selection"
        )
        set_module_status(
            data_dir,
            rid,
            "hca",
            STATUS_COMPLETED,
            started_at=started,
            finished_at=_now_iso(),
            elapsed_sec=elapsed,
            message=msg,
        )
        _tick(1.0, msg)
        return {
            "run_id": rid,
            "message": msg,
            "params": params,
            **summary,
        }
    except Exception as exc:
        set_module_status(
            data_dir,
            rid,
            "hca",
            STATUS_FAILED,
            started_at=started,
            finished_at=_now_iso(),
            message=str(exc),
        )
        raise


__all__ = [
    "DEFAULT_DISTANCE_THRESHOLD",
    "compute_hca_families",
    "ensure_hca_schema",
    "families_exist",
    "load_families",
    "load_family_members",
    "persist_hca_results",
    "run_hca_analysis",
]
