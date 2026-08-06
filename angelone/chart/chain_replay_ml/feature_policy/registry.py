"""Feature Policy Registry — load metadata + dependency graph."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from chain_replay_ml.dataset_builder.schema_registry import load_schema_registry, schema_registry_hash

from .metadata import FeaturePolicyMetadata, build_feature_policy_metadata, resolve_effective_warmup
from .types import FeatureCategory, FEATURE_POLICY_VERSION


class FeaturePolicyRegistry:
    """In-memory registry of policy metadata for all schema columns."""

    def __init__(self, metadata: dict[str, FeaturePolicyMetadata]) -> None:
        self._metadata = metadata
        self._children: dict[str, list[str]] = defaultdict(list)
        for name, meta in metadata.items():
            for dep in meta.dependencies:
                if dep in metadata or dep.startswith("__roll."):
                    self._children[dep].append(name)

    @property
    def features(self) -> dict[str, FeaturePolicyMetadata]:
        return dict(self._metadata)

    def get(self, name: str) -> FeaturePolicyMetadata | None:
        return self._metadata.get(name)

    def dependents(self, name: str) -> list[str]:
        return list(self._children.get(name, []))

    def by_category(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for name, meta in self._metadata.items():
            if meta.feature_category == FeatureCategory.METADATA:
                continue
            out[meta.feature_category.value].append(name)
        return dict(out)

    def classification_summary(self) -> dict[str, int]:
        counts = {k: 0 for k in (
            "raw", "rolling", "lookback", "cumulative", "derived", "target",
        )}
        for meta in self._metadata.values():
            cat = meta.feature_category.value
            if cat in counts:
                counts[cat] += 1
        return counts

    def rolling_features(self) -> list[FeaturePolicyMetadata]:
        return [
            m for m in self._metadata.values()
            if m.feature_category == FeatureCategory.ROLLING or m.reset_on_gap
        ]

    def validation_preview(
        self,
        *,
        sampling_interval_sec: float = 10.0,
        gap_max_sec: float = 20.0,
    ) -> dict[str, Any]:
        summary = self.classification_summary()
        rolling = [
            m for m in self._metadata.values()
            if m.feature_category in (FeatureCategory.ROLLING, FeatureCategory.DERIVED)
            and (m.effective_warmup_samples or m.intrinsic_warmup_samples)
        ]
        rolling = sorted(
            rolling,
            key=lambda m: m.effective_warmup_samples or m.intrinsic_warmup_samples,
        )
        warmup_rows = []
        for m in rolling[:12]:
            samples = m.effective_warmup_samples or m.intrinsic_warmup_samples
            warmup_rows.append({
                "name": m.name,
                "samples": samples,
                "potential_warmup_rows": max(0, samples - 1) if samples else 0,
                "warmup_time_sec": samples * sampling_interval_sec if samples else 0,
                "inherited": m.effective_warmup_inherited,
            })
        derived_derived = sum(
            1 for m in self._metadata.values()
            if m.feature_category == FeatureCategory.DERIVED
            and (m.effective_warmup_samples or m.effective_warmup_inherited)
        )
        return {
            "classification": summary,
            "rolling_policy": {
                "gap_max_sec": gap_max_sec,
                "warmup_mode": "sample_count",
                "reset_on_gap": True,
                "affected_rolling": summary.get("rolling", 0),
                "affected_derived_warmup": derived_derived,
            },
            "warmup_preview": warmup_rows,
        }


def load_feature_policy_registry(
    *,
    feature_names: list[str] | None = None,
) -> FeaturePolicyRegistry:
    reg = load_schema_registry()
    columns = reg.get("columns") or {}
    names = feature_names or [
        n for n, c in columns.items()
        if str(c.get("type") or "feature") in ("feature", "target")
    ]

    raw_meta: dict[str, FeaturePolicyMetadata] = {}
    for name in names:
        col = columns.get(name) or {"name": name, "type": "feature"}
        raw_meta[name] = build_feature_policy_metadata(name, col)

    # Virtual rolling anchors for derived EMA ratios
    for name, meta in list(raw_meta.items()):
        anchor = meta.policy_anchor
        if anchor and anchor not in raw_meta:
            m = anchor.split(".")
            period = int(m[2].replace("ema", "")) if len(m) > 2 else 0
            raw_meta[anchor] = FeaturePolicyMetadata(
                name=anchor,
                feature_category=FeatureCategory.ROLLING,
                lifecycle=meta.lifecycle,
                dependencies=("ltp",) if "ltp" in anchor else ("spot",),
                intrinsic_warmup_samples=period,
                rolling_type=meta.rolling_type,
                gap_sensitive=True,
                reset_on_gap=True,
                policy_version=meta.policy_version,
            )

    resolved: dict[str, FeaturePolicyMetadata] = {}
    for name, meta in raw_meta.items():
        resolved[name] = resolve_effective_warmup(meta, raw_meta)

    # Second pass with virtual anchors in registry
    for name in list(resolved.keys()):
        resolved[name] = resolve_effective_warmup(resolved[name], resolved)

    if feature_names:
        resolved = {k: v for k, v in resolved.items() if k in set(feature_names) or k.startswith("__roll.")}

    return FeaturePolicyRegistry(resolved)


def registry_version_info() -> dict[str, str]:
    return {
        "feature_policy_version": FEATURE_POLICY_VERSION,
        "feature_registry_version": schema_registry_hash(),
    }
