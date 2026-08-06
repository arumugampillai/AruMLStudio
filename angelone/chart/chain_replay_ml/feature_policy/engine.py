"""Feature Policy Engine — warm-up, readiness, gap reset, NULL propagation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .lifecycle import should_reset_on_gap, should_reset_on_session_start
from .registry import FeaturePolicyRegistry
from .types import DEFAULT_GAP_MAX_SEC, FeatureCategory, WarmupMode


@dataclass
class FeatureRuntimeState:
    samples_seen: int = 0
    ready: bool = False
    last_reset_reason: str | None = None


@dataclass
class EngineStats:
    samples_processed: int = 0
    gap_resets: int = 0
    session_resets: int = 0
    largest_gap_sec: float = 0.0
    derived_null_propagations: int = 0
    rolling_not_ready_outputs: int = 0


@dataclass
class FeaturePolicyEngine:
    """Standalone policy engine — no UI dependencies."""

    registry: FeaturePolicyRegistry
    sampling_interval_sec: float = 10.0
    gap_max_sec: float = DEFAULT_GAP_MAX_SEC
    reset_on_gap_enabled: bool = True
    _states: dict[str, FeatureRuntimeState] = field(default_factory=dict)
    _stats: EngineStats = field(default_factory=EngineStats)
    _last_ts: float | None = None
    _sample_index: int = 0
    _ready_cache: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in self.registry.features:
            self._states.setdefault(name, FeatureRuntimeState())

    def on_session_start(self) -> None:
        for name, meta in self.registry.features.items():
            if should_reset_on_session_start(meta.feature_category, meta.lifecycle) or name.startswith("__roll."):
                self._reset_feature(name, reason="session_start")
        self._stats.session_resets += 1
        self._last_ts = None
        self._sample_index = 0
        self._ready_cache.clear()

    def on_gap(self, gap_sec: float) -> None:
        if gap_sec <= self.gap_max_sec:
            return
        self._stats.largest_gap_sec = max(self._stats.largest_gap_sec, gap_sec)
        if not self.reset_on_gap_enabled:
            return
        for name, meta in self.registry.features.items():
            if should_reset_on_gap(meta.feature_category, reset_on_gap=meta.reset_on_gap) or name.startswith("__roll."):
                self._reset_feature(name, reason="gap")
        self._stats.gap_resets += 1

    def on_sample(self, ts: float) -> None:
        from .readiness_profiler import profiler_active, record_readiness_call

        t0 = time.perf_counter() if profiler_active() else 0.0
        if self._last_ts is not None:
            gap = ts - self._last_ts
            if gap > self.gap_max_sec:
                self.on_gap(gap)
        self._last_ts = ts
        self._sample_index += 1
        self._stats.samples_processed += 1
        self._ready_cache.clear()

        for name, meta in self.registry.features.items():
            if meta.feature_category == FeatureCategory.ROLLING or name.startswith("__roll."):
                self._tick_warmup(name, meta)
            elif meta.feature_category == FeatureCategory.LOOKBACK:
                self._tick_lookback(name, meta)
            elif (
                meta.feature_category == FeatureCategory.DERIVED
                and not meta.policy_anchor
                and not meta.effective_warmup_inherited
                and (meta.intrinsic_warmup_samples > 0 or meta.intrinsic_warmup_sec > 0)
            ):
                self._tick_lookback(name, meta)
        if profiler_active():
            record_readiness_call("on_sample", elapsed_sec=time.perf_counter() - t0)

    def is_ready(self, name: str) -> bool:
        from .readiness_profiler import profiler_active, record_readiness_call

        cached = self._ready_cache.get(name)
        if cached is not None:
            return cached
        t0 = time.perf_counter() if profiler_active() else 0.0
        try:
            result = self._is_ready_impl(name)
            self._ready_cache[name] = result
            return result
        finally:
            if profiler_active():
                record_readiness_call("is_ready", elapsed_sec=time.perf_counter() - t0)

    def _is_ready_impl(self, name: str) -> bool:
        from .readiness_profiler import profiler_active, record_readiness_call

        meta = self.registry.get(name)
        if profiler_active():
            record_readiness_call("registry.get")
        if not meta:
            return True
        if meta.feature_category in (FeatureCategory.RAW, FeatureCategory.METADATA, FeatureCategory.TARGET):
            return True
        if meta.feature_category == FeatureCategory.DERIVED:
            if meta.policy_anchor or meta.effective_warmup_inherited or (
                meta.intrinsic_warmup_samples == 0 and meta.intrinsic_warmup_sec == 0
            ):
                return self._deps_ready(meta)
            st = self._states.get(name)
            if st and not st.ready:
                return False
            return self._deps_ready(meta)
        if not self._deps_ready(meta):
            return False
        st = self._states.get(name)
        if not st:
            return True
        return st.ready

    def value_or_null(self, name: str, value: Any) -> Any:
        from .readiness_profiler import profiler_active, record_readiness_call

        t0 = time.perf_counter() if profiler_active() else 0.0
        try:
            if self.is_ready(name):
                return value
            meta = self.registry.get(name)
            if meta and meta.feature_category == FeatureCategory.DERIVED:
                self._stats.derived_null_propagations += 1
            elif meta and meta.feature_category == FeatureCategory.ROLLING:
                self._stats.rolling_not_ready_outputs += 1
            return None
        finally:
            if profiler_active():
                record_readiness_call("value_or_null", elapsed_sec=time.perf_counter() - t0)

    def readiness_snapshot(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name, st in self._states.items():
            meta = self.registry.get(name)
            if not meta or meta.feature_category == FeatureCategory.METADATA:
                continue
            out[name] = {
                "ready": self.is_ready(name),
                "samples_seen": st.samples_seen,
                "effective_warmup": meta.effective_warmup_samples,
                "last_reset": st.last_reset_reason,
            }
        return out

    def stats_dict(self) -> dict[str, Any]:
        return {
            "samples_processed": self._stats.samples_processed,
            "gap_resets": self._stats.gap_resets,
            "session_resets": self._stats.session_resets,
            "largest_gap_sec": self._stats.largest_gap_sec,
            "derived_null_propagations": self._stats.derived_null_propagations,
            "rolling_not_ready_outputs": self._stats.rolling_not_ready_outputs,
        }

    def _reset_feature(self, name: str, *, reason: str) -> None:
        st = self._states.setdefault(name, FeatureRuntimeState())
        st.samples_seen = 0
        st.ready = False
        st.last_reset_reason = reason

    def _tick_warmup(self, name: str, meta: Any) -> None:
        st = self._states.setdefault(name, FeatureRuntimeState())
        st.samples_seen += 1
        needed = meta.effective_warmup_samples or meta.intrinsic_warmup_samples
        if meta.warmup_mode == WarmupMode.TIME_SEC and meta.intrinsic_warmup_sec:
            needed = max(needed, int(meta.intrinsic_warmup_sec / max(self.sampling_interval_sec, 0.001)))
        if needed <= 0:
            st.ready = True
        else:
            st.ready = st.samples_seen >= needed

    def _tick_lookback(self, name: str, meta: Any) -> None:
        st = self._states.setdefault(name, FeatureRuntimeState())
        st.samples_seen += 1
        needed = meta.effective_warmup_samples
        if meta.intrinsic_warmup_sec and self.sampling_interval_sec > 0:
            needed = max(needed, int(meta.intrinsic_warmup_sec / self.sampling_interval_sec))
        if needed <= 0:
            st.ready = True
        else:
            st.ready = st.samples_seen >= needed

    def _deps_ready(self, meta: Any) -> bool:
        from .readiness_profiler import profiler_active, record_readiness_call

        t0 = time.perf_counter() if profiler_active() else 0.0
        try:
            for dep in meta.dependencies:
                if dep.startswith("__roll."):
                    st = self._states.get(dep)
                    if st and not st.ready:
                        return False
                    continue
                dep_meta = self.registry.get(dep)
                if not dep_meta:
                    continue
                if dep_meta.feature_category in (
                    FeatureCategory.RAW,
                    FeatureCategory.METADATA,
                    FeatureCategory.TARGET,
                ):
                    continue
                if not self.is_ready(dep):
                    return False
            return True
        finally:
            if profiler_active():
                record_readiness_call("_deps_ready", elapsed_sec=time.perf_counter() - t0)
