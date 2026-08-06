"""Build validation preview and post-build feature health."""

from __future__ import annotations

import math
from typing import Any

from .manifest import build_dataset_policy_manifest, build_report_from_manifest
from .registry import FeaturePolicyRegistry, load_feature_policy_registry
from .types import DEFAULT_GAP_MAX_SEC, FeatureCategory


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in ("", "nan", "none", "null"):
        return True
    return False


def _anchor_label(anchor: str | None) -> str | None:
    if not anchor:
        return None
    parts = str(anchor).split(".")
    if len(parts) >= 3 and parts[2].startswith("ema"):
        return f"EMA{parts[2].replace('ema', '')}"
    return anchor.replace("__roll.", "")


def _is_inherited_meta(meta: Any) -> bool:
    if meta.effective_warmup_inherited or meta.policy_anchor:
        return True
    if meta.feature_category == FeatureCategory.DERIVED and (
        meta.effective_warmup_samples or meta.intrinsic_warmup_samples
    ):
        return True
    return False


def build_validation_preview(
    feature_names: list[str],
    *,
    sampling_interval_sec: float = 10.0,
    estimated_rows: int | None = None,
    estimated_sessions: int | None = None,
    gap_max_sec: float = DEFAULT_GAP_MAX_SEC,
) -> dict[str, Any]:
    """Pre-build validation snapshot: classification, warm-up rows/time, checks."""
    names = list(dict.fromkeys(feature_names))
    reg = load_feature_policy_registry(feature_names=names if names else None)
    preview = reg.validation_preview(
        sampling_interval_sec=sampling_interval_sec,
        gap_max_sec=gap_max_sec,
    )
    warmup_rows = []
    max_warmup = 0
    max_warmup_name = ""
    inherited_features = 0
    for meta in reg.features.values():
        if meta.name.startswith("__roll."):
            continue
        if _is_inherited_meta(meta):
            inherited_features += 1
        samples = meta.effective_warmup_samples or meta.intrinsic_warmup_samples
        if not samples:
            continue
        if samples >= max_warmup:
            max_warmup = samples
            max_warmup_name = meta.name
        inh_from = _anchor_label(meta.policy_anchor)
        warmup_rows.append({
            "name": meta.name,
            "samples": samples,
            "potential_warmup_rows": max(0, samples - 1),
            "warmup_time_sec": samples * sampling_interval_sec,
            "inherited": _is_inherited_meta(meta),
            "inherited_from": inh_from,
            "category": meta.feature_category.value,
        })
    warmup_rows.sort(key=lambda r: (-int(r["samples"]), r["name"]))

    warmup_per_session = max(0, max_warmup - 1) if max_warmup else 0
    sessions = estimated_sessions
    if sessions is None and estimated_rows and warmup_per_session:
        sessions = max(1, int(estimated_rows / max(1, estimated_rows // max(1, warmup_per_session * 10))))

    est_warmup_rows = None
    if estimated_rows is not None and warmup_per_session:
        if sessions:
            est_warmup_rows = warmup_per_session * sessions
        else:
            est_warmup_rows = min(estimated_rows, warmup_per_session)

    checks = [
        {
            "id": "feature_count",
            "label": f"{len(names)} features selected",
            "status": "pass" if names else "fail",
        },
        {
            "id": "rolling_policy",
            "label": f"Gap reset > {gap_max_sec:g}s on rolling features",
            "status": "pass",
        },
        {
            "id": "warmup_budget",
            "label": (
                f"Max warm-up: {max_warmup_name or '—'} ({max_warmup} samples · "
                f"~{max_warmup * sampling_interval_sec:.0f}s)"
                if max_warmup
                else "No rolling warm-up required"
            ),
            "status": "pass",
        },
    ]
    if estimated_rows is not None:
        checks.append({
            "id": "estimated_rows",
            "label": f"~{int(estimated_rows):,} estimated output rows",
            "status": "pass" if estimated_rows > 0 else "fail",
        })
    if est_warmup_rows is not None:
        checks.append({
            "id": "warmup_rows",
            "label": f"~{int(est_warmup_rows):,} warm-up rows across sessions",
            "status": "pass",
        })

    return {
        **preview,
        "feature_count": len(names),
        "max_warmup_samples": max_warmup,
        "max_warmup_feature": max_warmup_name,
        "warmup_rows_per_session": warmup_per_session,
        "estimated_warmup_rows": est_warmup_rows,
        "estimated_warmup_time_sec": max_warmup * sampling_interval_sec if max_warmup else 0,
        "longest_ready_time": (
            f"≈{(max_warmup * sampling_interval_sec) / 60:.1f} min"
            if max_warmup * sampling_interval_sec >= 60
            else f"≈{max_warmup * sampling_interval_sec:.0f} sec"
        ) if max_warmup else "—",
        "inherited_features": inherited_features,
        "warmup_preview": warmup_rows[:20],
        "checks": checks,
        "gap_max_sec": gap_max_sec,
        "sampling_interval_sec": sampling_interval_sec,
    }


def compute_feature_health_from_rows(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    *,
    registry: FeaturePolicyRegistry | None = None,
    session_key: str = "trading_day",
) -> list[dict[str, Any]]:
    """Estimate per-feature ready / warm-up / missing % from built rows."""
    if not rows or not feature_names:
        return []
    if registry is None:
        registry = load_feature_policy_registry(feature_names=feature_names)

    names = [n for n in feature_names if not n.startswith("__roll.")]
    out: list[dict[str, Any]] = []

    def session_leading_nulls(series: list[Any]) -> int:
        n = 0
        for v in series:
            if _is_null(v):
                n += 1
            else:
                break
        return n

    # group indices by session
    session_indices: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        key = str(row.get(session_key) or row.get("date") or "_all")
        session_indices.setdefault(key, []).append(i)

    for name in names:
        meta = registry.get(name)
        cat = meta.feature_category.value if meta else "raw"
        total = len(rows)
        missing = sum(1 for r in rows if _is_null(r.get(name)))
        ready = total - missing
        warmup_rows = 0
        if cat in (FeatureCategory.ROLLING.value, FeatureCategory.DERIVED.value, FeatureCategory.LOOKBACK.value):
            for indices in session_indices.values():
                series = [rows[i].get(name) for i in indices]
                warmup_rows += session_leading_nulls(series)
        policy_warmup = 0
        if meta:
            policy_warmup = meta.effective_warmup_samples or meta.intrinsic_warmup_samples
        out.append({
            "name": name,
            "rows": total,
            "ready_pct": round(ready / max(total, 1) * 100.0, 2),
            "warmup_pct": round(warmup_rows / max(total, 1) * 100.0, 2),
            "warmup_rows": warmup_rows,
            "missing_pct": round(missing / max(total, 1) * 100.0, 2),
            "gap_reset_pct": 0.0,
            "gap_reset_count": 0,
            "policy_warmup_samples": policy_warmup,
            "category": cat,
        })
    out.sort(key=lambda r: (-float(r.get("missing_pct") or 0), r["name"]))
    return out


def finalize_build_policy_manifest(
    feature_names: list[str],
    *,
    sampling_interval_sec: float,
    gap_max_sec: float = DEFAULT_GAP_MAX_SEC,
    rows: list[dict[str, Any]] | None = None,
    build_stats: dict[str, Any] | None = None,
    health_sample_limit: int = 50_000,
) -> dict[str, Any]:
    """Build frozen policy manifest + health for dataset metadata."""
    names = list(dict.fromkeys(feature_names))
    reg = load_feature_policy_registry(feature_names=names if names else None)
    health: list[dict[str, Any]] = []
    stats = dict(build_stats or {})
    if rows:
        sample = rows[:health_sample_limit]
        health = compute_feature_health_from_rows(sample, names, registry=reg)
        stats.setdefault("rows_sampled", len(sample))
        if health:
            stats["avg_ready_pct"] = round(
                sum(h.get("ready_pct", 0) for h in health) / len(health), 2,
            )
            worst = max(health, key=lambda h: float(h.get("missing_pct") or 0))
            stats["worst_missing_feature"] = worst.get("name")
            stats["worst_missing_pct"] = worst.get("missing_pct")
    return build_dataset_policy_manifest(
        reg,
        sampling_interval_sec=sampling_interval_sec,
        gap_max_sec=gap_max_sec,
        selected_features=names,
        build_stats=stats,
        health_summary=health,
    )


def sample_rows_from_sqlite(
    conn: Any,
    columns: list[str],
    *,
    limit: int = 50_000,
) -> list[dict[str, Any]]:
    """Sample built rows from master SQLite for health estimation."""
    meta_cols = ["trading_day", "timestamp", "token"]
    feat_cols = [c for c in columns if c not in meta_cols]
    use_cols = [c for c in meta_cols if c] + feat_cols[:100]
    if not use_cols:
        return []
    col_sql = ", ".join(f'"{c}"' for c in use_cols)
    cur = conn.execute(
        f"SELECT {col_sql} FROM samples ORDER BY trading_day, timestamp LIMIT ?",
        (int(limit),),
    )
    if not cur.description:
        return []
    keys = [d[0] for d in cur.description]
    return [dict(zip(keys, row)) for row in cur.fetchall()]


def build_policy_report(manifest: dict[str, Any]) -> dict[str, Any]:
    """UI-friendly build report from persisted manifest."""
    base = build_report_from_manifest(manifest)
    health = manifest.get("feature_health") or []
    base["feature_health"] = health
    base["classification"] = manifest.get("classification") or {}
    base["selected_features"] = manifest.get("selected_features") or []
    if health:
        base["health_summary"] = {
            "features_tracked": len(health),
            "avg_ready_pct": manifest.get("build_stats", {}).get("avg_ready_pct"),
            "worst_missing_feature": manifest.get("build_stats", {}).get("worst_missing_feature"),
            "worst_missing_pct": manifest.get("build_stats", {}).get("worst_missing_pct"),
        }
    return base
