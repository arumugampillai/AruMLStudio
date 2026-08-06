"""Stage / function / controller profiler for dataset builds (perf_counter-based)."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Iterator, TypeVar

_T = TypeVar("_T")

_lock = threading.Lock()
_active: BuildProfiler | None = None


@dataclass
class ProfileStat:
    total_sec: float = 0.0
    call_count: int = 0
    max_sec: float = 0.0
    rows: int = 0

    def record(self, duration_sec: float, *, rows: int = 0) -> None:
        d = max(0.0, float(duration_sec))
        self.total_sec += d
        self.call_count += 1
        if d > self.max_sec:
            self.max_sec = d
        if rows > 0:
            self.rows += int(rows)

    def merge(self, other: ProfileStat) -> None:
        self.total_sec += other.total_sec
        self.call_count += other.call_count
        self.rows += other.rows
        if other.max_sec > self.max_sec:
            self.max_sec = other.max_sec

    def avg_sec(self) -> float:
        if self.call_count <= 0:
            return 0.0
        return self.total_sec / self.call_count

    def rows_per_sec(self) -> float | None:
        if self.rows <= 0 or self.total_sec <= 0:
            return None
        return self.rows / self.total_sec

    def pct_of(self, total_sec: float) -> float | None:
        if total_sec <= 0 or self.total_sec <= 0:
            return None
        return 100.0 * self.total_sec / total_sec


def _stat_row(
    name: str,
    stat: ProfileStat,
    *,
    total_sec: float,
    category: str,
) -> dict[str, Any]:
    rps = stat.rows_per_sec()
    return {
        "name": name,
        "category": category,
        "total_sec": round(stat.total_sec, 6),
        "call_count": stat.call_count,
        "avg_ms": round(stat.avg_sec() * 1000.0, 4),
        "max_ms": round(stat.max_sec * 1000.0, 4),
        "rows": stat.rows,
        "rows_per_sec": round(rps, 1) if rps is not None else None,
        "pct": round(stat.pct_of(total_sec) or 0.0, 2) if total_sec > 0 else None,
    }


# Registry group id → profiler family label (rollup section).
GROUP_TO_FAMILY: dict[str, str] = {
    "price": "Price & Returns",
    "dgt_reiv": "DGT",
    "ratio": "Price & Returns",
    "greeks": "Greeks",
    "iv": "IV",
    "iv_zscore": "IV",
    "iv_ema_ratio": "IV EMA Ratio",
    "oi": "Chain Maps",
    "volume": "Chain Maps",
    "momentum": "RV",
    "sharp_momentum": "Spot Momentum",
    "spot_hl": "Spot HL",
    "time": "Price & Returns",
    "moneyness": "Price & Returns",
    "ltp_to_spot": "Price & Returns",
    "ltp_to_others": "Price & Returns",
    "spot_and_other_ratio": "RV",
    "atm_straddle": "Chain Maps",
    "atm6_ltp": "Chain Maps",
    "chain": "Chain Maps",
    "historical": "Price & Returns",
    "advanced": "Advanced Features",
}

# Function keys → family for rollup when group loop timing is thin.
FUNCTION_TO_FAMILY: dict[str, str] = {
    "function.extract_timeline_features": "Price & Returns",
    "function.update_token_ltp_controllers": "Price & Returns",
    "function.update_token_rv_controllers": "RV",
    "function.spot_controllers.update": "RV",
    "function.enrich_spot_momentum_registry": "Spot Momentum",
    "function.enrich_dataset_features": "DGT",
    "function.enrich_with_chain_maps": "Chain Maps",
    "function.enrich_sharp_momentum": "Spot Momentum",
    "function.enrich_iv_zscore": "IV",
    "function.enrich_iv_ema_ratio": "IV EMA Ratio",
    "function.enrich_spot_ratio_moneyness": "Spot and Other Ratio",
    "function.enrich_advanced_composites": "Advanced Features",
    "function.enrich_spot_hl_ratio": "Spot HL",
    "function.enrich_spot_hl_composite": "Spot HL",
    "function.enrich_current_to_atm6_flow": "Chain Maps",
    "function.build_feature_raw_for_row": "Feature Generation",
    "function.pick_features_from_row": "Feature Generation",
    "stage.prep.chain_maps": "Chain Maps",
    "stage.prep.sharp_momentum": "Spot Momentum",
}


@dataclass
class BuildProfiler:
    """Accumulates perf_counter timings for one build job (may merge per-day shards)."""

    _stats: dict[str, ProfileStat] = field(default_factory=dict)
    _build_start: float | None = None
    _build_end: float | None = None
    _total_rows: int = 0

    def start_build(self) -> None:
        self._build_start = perf_counter()

    def finish_build(self, *, total_rows: int = 0) -> None:
        self._build_end = perf_counter()
        self._total_rows = max(0, int(total_rows))

    @property
    def total_build_sec(self) -> float:
        if self._build_start is None:
            return 0.0
        end = self._build_end if self._build_end is not None else perf_counter()
        return max(0.0, end - self._build_start)

    def record(self, key: str, duration_sec: float, *, rows: int = 0) -> None:
        stat = self._stats.setdefault(key, ProfileStat())
        stat.record(duration_sec, rows=rows)

    @contextmanager
    def time(self, key: str, *, rows: int = 0) -> Iterator[None]:
        t0 = perf_counter()
        try:
            yield
        finally:
            self.record(key, perf_counter() - t0, rows=rows)

    def merge(self, other: BuildProfiler) -> None:
        for key, stat in other._stats.items():
            self._stats.setdefault(key, ProfileStat()).merge(stat)
        self._total_rows += other._total_rows

    def _sorted_entries(self, prefix: str, category: str) -> list[dict[str, Any]]:
        total = self.total_build_sec
        items = [
            (key, stat)
            for key, stat in self._stats.items()
            if key.startswith(prefix)
        ]
        items.sort(key=lambda kv: kv[1].total_sec, reverse=True)
        return [_stat_row(key, stat, total_sec=total, category=category) for key, stat in items]

    def _rollup_families(self) -> list[dict[str, Any]]:
        totals: dict[str, ProfileStat] = {}
        for key, stat in self._stats.items():
            family: str | None = None
            if key.startswith("group."):
                gid = key[len("group.") :]
                family = GROUP_TO_FAMILY.get(gid)
            elif key.startswith("function.") or key.startswith("stage.prep."):
                family = FUNCTION_TO_FAMILY.get(key)
            elif key.startswith("controller."):
                ctrl = key[len("controller.") :]
                if ctrl.startswith("token.ema") or ctrl.startswith("spot.ema"):
                    family = "RV"
                elif "rv" in ctrl.lower():
                    family = "RV"
                elif ctrl.startswith("token.iv") or ctrl == "iv_history":
                    family = "IV"
                elif ctrl == "dgt" or ctrl == "roll":
                    family = "DGT" if ctrl == "dgt" else "DGT"
                elif ctrl.startswith("spot.hl") or ctrl == "spot_hl":
                    family = "Spot HL"
                elif ctrl == "spot.momentum":
                    family = "Spot Momentum"
            if not family:
                continue
            totals.setdefault(family, ProfileStat()).merge(stat)

        total_sec = self.total_build_sec
        ranked = sorted(totals.items(), key=lambda kv: kv[1].total_sec, reverse=True)
        return [
            _stat_row(f"family.{name}", stat, total_sec=total_sec, category="family")
            for name, stat in ranked
        ]

    def to_report(self) -> dict[str, Any]:
        total = self.total_build_sec
        stages = self._sorted_entries("stage.", "stage")
        functions = self._sorted_entries("function.", "function")
        controllers = self._sorted_entries("controller.", "controller")
        groups = self._sorted_entries("group.", "group")
        families = self._rollup_families()

        all_entries = stages + functions + controllers + groups + families
        all_entries.sort(key=lambda e: float(e.get("total_sec") or 0), reverse=True)

        return {
            "total_build_sec": round(total, 4),
            "total_rows": self._total_rows,
            "build_rows_per_sec": (
                round(self._total_rows / total, 1) if total > 0 and self._total_rows > 0 else None
            ),
            "stages": stages,
            "functions": functions,
            "controllers": controllers,
            "feature_groups": groups,
            "feature_families": families,
            "ranked": all_entries,
        }


def profiler_active() -> bool:
    return get_profiler() is not None


def get_profiler() -> BuildProfiler | None:
    return _active


def set_profiler(profiler: BuildProfiler | None) -> None:
    global _active
    with _lock:
        _active = profiler


@contextmanager
def build_profiler_session(*, enabled: bool = True) -> Iterator[BuildProfiler | None]:
    if not enabled:
        yield None
        return
    prof = BuildProfiler()
    prof.start_build()
    set_profiler(prof)
    try:
        yield prof
    finally:
        set_profiler(None)


@contextmanager
def profile_block(key: str, *, rows: int = 0) -> Iterator[None]:
    prof = get_profiler()
    if prof is None:
        yield
        return
    t0 = perf_counter()
    try:
        yield
    finally:
        prof.record(key, perf_counter() - t0, rows=rows)


def profile_call(key: str, fn: Callable[[], _T], *, rows: int = 0) -> _T:
    prof = get_profiler()
    if prof is None:
        return fn()
    t0 = perf_counter()
    try:
        return fn()
    finally:
        prof.record(key, perf_counter() - t0, rows=rows)
