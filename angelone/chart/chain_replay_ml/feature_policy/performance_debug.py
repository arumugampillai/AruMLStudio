"""Performance / debug profiling levels for dataset and simulator builds."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any


class PerformanceDebugLevel(str, Enum):
    """Controls how much profiling instrumentation runs during builds."""

    OFF = "off"  # Production — near-zero overhead
    BASIC = "basic"  # Pipeline wall-clock timings only
    FULL = "full"  # All profilers, comparisons, cProfile, diagnostics

    @classmethod
    def from_value(cls, value: Any) -> PerformanceDebugLevel:
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.OFF
        text = str(value).strip().lower()
        for member in cls:
            if member.value == text or member.name.lower() == text:
                return member
        if value is True:
            return cls.FULL
        return cls.OFF

    def _rank(self) -> int:
        return {PerformanceDebugLevel.OFF: 0, PerformanceDebugLevel.BASIC: 1, PerformanceDebugLevel.FULL: 2}[self]

    def at_least(self, minimum: PerformanceDebugLevel) -> PerformanceDebugLevel:
        return self if self._rank() >= minimum._rank() else minimum

    def ui_label(self) -> str:
        return {
            PerformanceDebugLevel.OFF: "Production",
            PerformanceDebugLevel.BASIC: "Basic",
            PerformanceDebugLevel.FULL: "Full Debug",
        }[self]


@dataclass(frozen=True)
class PerformanceDebugConfig:
    """Central gate for all profiling / diagnostic collection."""

    level: PerformanceDebugLevel = PerformanceDebugLevel.OFF

    @classmethod
    def resolve(
        cls,
        level: PerformanceDebugLevel | str | Any | None = None,
        *,
        gap_profile: bool = False,
        readiness_profile: bool = False,
        config: PerformanceDebugConfig | None = None,
    ) -> PerformanceDebugConfig:
        if isinstance(config, cls):
            return config
        resolved = PerformanceDebugLevel.from_value(level)
        if readiness_profile:
            resolved = resolved.at_least(PerformanceDebugLevel.FULL)
        elif gap_profile:
            resolved = resolved.at_least(PerformanceDebugLevel.FULL)
        if os.getenv("GAP_POLICY_PROFILE", "") == "1":
            resolved = resolved.at_least(PerformanceDebugLevel.FULL)
        return cls(level=resolved)

    def enabled(self) -> bool:
        return self.level != PerformanceDebugLevel.OFF

    def collect_pipeline_timings(self) -> bool:
        return self.level._rank() >= PerformanceDebugLevel.BASIC._rank()

    def collect_build_profiler(self) -> bool:
        """Stage / function / controller perf_counter profiler (master build)."""
        return self.level._rank() >= PerformanceDebugLevel.BASIC._rank()

    def collect_gap_profile(self) -> bool:
        return self.level == PerformanceDebugLevel.FULL

    def collect_cprofile(self) -> bool:
        return self.level == PerformanceDebugLevel.FULL

    def collect_readiness_profile(self) -> bool:
        return self.level == PerformanceDebugLevel.FULL

    def run_gap_pass_comparison(self, *, explicit: bool = False, gap_parity: bool = False) -> bool:
        if self.level != PerformanceDebugLevel.FULL:
            return False
        return bool(explicit or gap_parity)

    def run_cache_benchmark(self) -> bool:
        return self.level == PerformanceDebugLevel.FULL

    def run_lookback_dual_pass(self, *, explicit: bool = False) -> bool:
        return self.level == PerformanceDebugLevel.FULL and bool(explicit)

    def collect_function_diff(self) -> bool:
        return self.level == PerformanceDebugLevel.FULL

    def show_replay_pipeline(self) -> bool:
        return self.collect_pipeline_timings()

    def show_gap_policy_summary(self) -> bool:
        return self.level == PerformanceDebugLevel.FULL

    def show_gap_policy_full_detail(self) -> bool:
        return self.level == PerformanceDebugLevel.FULL

    def show_full_diagnostics(self) -> bool:
        return self.level == PerformanceDebugLevel.FULL
