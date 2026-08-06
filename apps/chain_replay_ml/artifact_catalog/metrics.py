"""Research productivity metrics over Catalog + Timeline."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from .store import ArtifactCatalogStore
from .types import ResearchMetrics


def _parse_ts(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def compute_research_metrics(store: ArtifactCatalogStore) -> ResearchMetrics:
    """Aggregate research-ops metrics from catalogued artifacts."""
    experiments = store.list_by_type("experiment")
    completed_exps = [e for e in experiments if e.status in ("completed", "failed", "running")]
    # Prefer completed for "run" count.
    experiments_run = sum(1 for e in experiments if e.status == "completed")

    label_counts: Counter[str] = Counter()
    for r in store.list_by_type("training"):
        strat = r.metadata.get("strategy") or r.metadata.get("label_strategy")
        if strat:
            label_counts[str(strat)] += 1
    for e in experiments:
        strat = (e.metadata.get("actions") or {}).get("label_strategy") if isinstance(
            e.metadata.get("actions"), dict
        ) else None
        if not strat:
            strat = e.metadata.get("label_strategy")
        if strat:
            label_counts[str(strat)] += 1
    best_label = label_counts.most_common(1)[0][0] if label_counts else None

    feature_sets: Counter[str] = Counter()
    for m in store.list_by_type("model"):
        fs = m.metadata.get("feature_set_id") or m.metadata.get("features_hash")
        if not fs and isinstance(m.metadata.get("feature_count"), int):
            fs = f"n={m.metadata['feature_count']}"
        if fs:
            feature_sets[str(fs)] += 1
    most_reused = feature_sets.most_common(1)[0][0] if feature_sets else None

    # Avg time dataset→model using parent training/dataset and model created_at.
    latencies: list[float] = []
    for m in store.list_by_type("model"):
        m_ts = _parse_ts(m.created_at)
        if m_ts is None:
            continue
        for p in m.parent_artifact_uris:
            parent = store.get(p)
            if parent is None:
                continue
            if parent.artifact_type not in ("training", "prediction", "master"):
                continue
            p_ts = _parse_ts(parent.created_at)
            if p_ts is None:
                continue
            delta = (m_ts - p_ts).total_seconds()
            if delta >= 0:
                latencies.append(delta)
    avg_latency = (sum(latencies) / len(latencies)) if latencies else None

    # Improvement vs previous: mean of metadata.improvement_pct on completed experiments.
    improvements: list[float] = []
    for e in experiments:
        if e.status != "completed":
            continue
        val = e.metadata.get("improvement_pct")
        if val is None and isinstance(e.metadata.get("result"), dict):
            val = e.metadata["result"].get("improvement_pct")
        try:
            if val is not None:
                improvements.append(float(val))
        except (TypeError, ValueError):
            pass
    improvement = (
        improvements[-1] if improvements else None
    )  # latest completed experiment lift

    extras: dict[str, Any] = {
        "label_strategy_counts": dict(label_counts),
        "feature_set_counts": dict(feature_sets),
        "experiments_total": len(experiments),
        "experiments_completed": experiments_run,
        "models": len(store.list_by_type("model")),
        "training_datasets": len(store.list_by_type("training")),
    }
    return ResearchMetrics(
        experiments_run=experiments_run,
        best_label_strategy=best_label,
        most_reused_feature_set=most_reused,
        avg_dataset_to_model_sec=avg_latency,
        improvement_vs_previous=improvement,
        extras=extras,
    )


def format_evidence_summary(metrics: ResearchMetrics, *, limit: int = 6) -> str:
    """Human-readable evidence block for Planner experiment contracts."""
    lines = [
        f"Experiments completed: {metrics.experiments_run}",
    ]
    if metrics.best_label_strategy:
        lines.append(f"Best-performing / most-used label strategy: {metrics.best_label_strategy}")
    counts = (metrics.extras or {}).get("label_strategy_counts") or {}
    if isinstance(counts, dict) and counts:
        top = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:limit]
        lines.append("Recent label strategy usage:")
        for name, n in top:
            lines.append(f"  {name} → n={n}")
    if metrics.improvement_vs_previous is not None:
        lines.append(f"Latest experiment improvement_pct: {metrics.improvement_vs_previous}")
    if metrics.avg_dataset_to_model_sec is not None:
        lines.append(
            f"Avg dataset→model latency: {metrics.avg_dataset_to_model_sec:.1f}s"
        )
    return "\n".join(lines)
