"""Build static feature policy metadata from registry + parity specs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from chain_replay_ml.dataset_builder.feature_grid_policy import (
    FeatureComputationKind,
    build_feature_parity_spec,
)
from chain_replay_ml.dataset_builder.schema_feature_meta import infer_depends_on

from .types import (
    DEFAULT_GAP_MAX_SEC,
    FEATURE_POLICY_VERSION,
    FeatureCategory,
    FeatureLifecycle,
    RollingType,
    WarmupMode,
)

_EMA_IN_NAME = re.compile(r"(?:ltp_ema|spot_ema|iv_ema|ema)(\d+)", re.I)
_CALENDAR_SEC = re.compile(r"_(\d+)(?:s|sec|m|min)\b", re.I)
_WARMUP_BARS = re.compile(r"^(\d+)\s+bars?$", re.I)
_WARMUP_SEC = re.compile(r"^(\d+)\s+seconds?$", re.I)

# Ratio features → intrinsic rolling warm-up (matches extended_features)
_EMA_RATIO_WARMUP: dict[str, int] = {
    "ltp_ema9_to_ltp_ratio": 9,
    "ltp_ema20_to_ltp_ratio": 20,
    "ltp_ema50_to_ltp_ratio": 50,
    "ltp_ema100_to_ltp_ratio": 100,
    "ltp_ema200_to_ltp_ratio": 200,
    "ltp_ema9_to_spot_ratio": 9,
    "ltp_ema20_to_spot_ratio": 20,
    "ltp_ema50_to_spot_ratio": 50,
    "ltp_ema100_to_spot_ratio": 100,
    "ltp_ema200_to_spot_ratio": 200,
    "ltp_std20_to_ltp_ratio": 20,
    "ltp_std20_to_spot_ratio": 20,
}

for _p in (9, 20, 50, 100, 200, 300):
    _EMA_RATIO_WARMUP[f"ltp_ema{_p}_to_spot_ratio_x_iv_ema{_p}"] = _p
    _EMA_RATIO_WARMUP[f"spot_to_ltp_ratio_x_iv_ema{_p}"] = _p
    _EMA_RATIO_WARMUP[f"iv_ema{_p}_to_ltp_ratio"] = _p
    _EMA_RATIO_WARMUP[f"iv_ema{_p}_to_spot_ratio"] = _p
    _EMA_RATIO_WARMUP[f"spot_to_ltp_ratio_x_iv_ema{_p}_x_moneyness"] = _p
    _EMA_RATIO_WARMUP[f"spot_ema{_p}_to_ltp_ratio_x_moneyness"] = _p
    _EMA_RATIO_WARMUP[f"ltp_ema{_p}_to_spot_ratio_x_moneyness"] = _p

_EMA_RATIO_WARMUP["spot_ema300_to_ltp_ratio"] = 300

# Weighted multi-EMA blends — bottleneck is the longest EMA period (200).
_WEIGHTED_EMA_WARMUP: dict[str, int] = {
    "weighted_ltp_ema_to_ltp_ratio": 200,
    "weighted_spot_ema_to_ltp_ratio": 200,
}

_CUMULATIVE_NAMES = frozenset({
    "vwap",
    "session_volume",
    "cumulative_volume",
    "day_volume",
})

# Exchange as-of ATP levels (SNAP_QUOTE average_traded_price). These are RAW
# tape fields, not session-cumulative rebuilds. The naive ``"vwap" in name``
# heuristic must not classify them as CUMULATIVE — that lifecycle never
# warm-ticks, so readiness stays False and Master writes 100% NULL.
_RAW_EXCHANGE_ATP_VWAP_NAMES = frozenset({
    "option_vwap",
    "futures_vwap",
})


@dataclass(frozen=True)
class FeaturePolicyMetadata:
    name: str
    feature_category: FeatureCategory
    lifecycle: FeatureLifecycle
    dependencies: tuple[str, ...]
    intrinsic_warmup_samples: int = 0
    intrinsic_warmup_sec: int = 0
    warmup_mode: WarmupMode = WarmupMode.SAMPLE_COUNT
    rolling_type: RollingType | None = None
    gap_sensitive: bool = False
    reset_on_gap: bool = False
    formula_version: str = "1"
    policy_version: str = "1"
    group_id: str = ""
    col_type: str = "feature"
    parity_kind: str = ""
    effective_warmup_samples: int = 0
    effective_warmup_inherited: bool = False
    policy_anchor: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature_category": self.feature_category.value,
            "lifecycle": self.lifecycle.value,
            "dependencies": list(self.dependencies),
            "intrinsic_warmup_samples": self.intrinsic_warmup_samples,
            "intrinsic_warmup_sec": self.intrinsic_warmup_sec,
            "warmup_mode": self.warmup_mode.value,
            "rolling_type": self.rolling_type.value if self.rolling_type else None,
            "gap_sensitive": self.gap_sensitive,
            "reset_on_gap": self.reset_on_gap,
            "formula_version": self.formula_version,
            "policy_version": self.policy_version,
            "group_id": self.group_id,
            "col_type": self.col_type,
            "parity_kind": self.parity_kind,
            "effective_warmup_samples": self.effective_warmup_samples,
            "effective_warmup_inherited": self.effective_warmup_inherited,
            "policy_anchor": self.policy_anchor,
        }


def parse_warmup_string(warmup: str) -> tuple[int, int, WarmupMode]:
    w = str(warmup or "").strip()
    m = _WARMUP_BARS.match(w)
    if m:
        return int(m.group(1)), 0, WarmupMode.SAMPLE_COUNT
    m = _WARMUP_SEC.match(w)
    if m:
        return 0, int(m.group(1)), WarmupMode.TIME_SEC
    return 0, 0, WarmupMode.SAMPLE_COUNT


def _infer_rolling_type(name: str) -> RollingType:
    n = name.lower()
    if "ema" in n:
        return RollingType.EMA
    if "sma" in n:
        return RollingType.SMA
    if "std" in n:
        return RollingType.STD
    if "atr" in n:
        return RollingType.ATR
    if "rsi" in n:
        return RollingType.RSI
    if "bollinger" in n or "bb_" in n:
        return RollingType.BOLLINGER
    if "_rv_" in n or n.endswith("_rv_ratio"):
        return RollingType.RV
    if "zscore" in n:
        return RollingType.ZSCORE
    return RollingType.OTHER


def _policy_anchor_name(name: str) -> str | None:
    if name in _WEIGHTED_EMA_WARMUP:
        base = "spot" if name.startswith("weighted_spot_") else "ltp"
        return f"__roll.{base}.ema{_WEIGHTED_EMA_WARMUP[name]}"
    iv_m = re.search(r"iv_ema(\d+)", name, re.I)
    if iv_m and ("iv_ema" in name.lower()):
        # IV-EMA ratios / products bottleneck on the IV EMA controller.
        return f"__roll.iv.ema{iv_m.group(1)}"
    if name in _EMA_RATIO_WARMUP:
        m = _EMA_IN_NAME.search(name)
        if m:
            base = "spot" if name.startswith("spot_") else "ltp"
            return f"__roll.{base}.ema{m.group(1)}"
    m = _EMA_IN_NAME.search(name)
    if m and ("ratio" in name or "ema" in name.lower()):
        base = "spot" if "spot_ema" in name else "ltp"
        return f"__roll.{base}.ema{m.group(1)}"
    return None


def _augment_dependencies(name: str, deps: list[str], category: FeatureCategory) -> tuple[str, ...]:
    out = list(dict.fromkeys(str(d) for d in deps if d and not d.startswith("feature_grid")))
    anchor = _policy_anchor_name(name)
    if anchor and anchor not in out:
        if category == FeatureCategory.DERIVED:
            out.append(anchor)
    return tuple(out)


def infer_feature_category(
    name: str,
    *,
    col_type: str = "feature",
    group_id: str = "",
    parity_kind: FeatureComputationKind,
    depends_on: list[str] | None = None,
) -> FeatureCategory:
    if col_type == "target":
        return FeatureCategory.TARGET
    if col_type == "metadata":
        return FeatureCategory.METADATA

    n = name.lower()
    if name in _RAW_EXCHANGE_ATP_VWAP_NAMES or n in _RAW_EXCHANGE_ATP_VWAP_NAMES:
        return FeatureCategory.RAW
    # Historic NIFTY bar EMAs: precomputed as-of levels (not session tick rollings).
    if group_id == "historic_spot_ema" or re.match(
        r"^spot_(?:1m|3m|5m|15m)_ema\d+$", name, re.I
    ):
        return FeatureCategory.RAW
    if n in _CUMULATIVE_NAMES or (
        "vwap" in n and n not in _RAW_EXCHANGE_ATP_VWAP_NAMES
    ) or (n.endswith("_volume") and "session" in n):
        return FeatureCategory.CUMULATIVE
    if name in _EMA_RATIO_WARMUP or (
        depends_on and len(depends_on) > 1 and ("ratio" in n or "_to_" in n)
    ):
        if parity_kind == FeatureComputationKind.GRID_BAR and _EMA_IN_NAME.search(name):
            return FeatureCategory.DERIVED
        if "ratio" in n or "_to_" in n:
            return FeatureCategory.DERIVED
    if parity_kind == FeatureComputationKind.GRID_BAR:
        return FeatureCategory.ROLLING
    if parity_kind == FeatureComputationKind.CALENDAR_SEC:
        return FeatureCategory.LOOKBACK
    if parity_kind == FeatureComputationKind.STATIC:
        if depends_on and len([d for d in depends_on if d not in ("timestamp", "token")]) > 0:
            if n in ("ltp", "spot", "bid", "ask", "oi", "volume", "delta", "gamma", "theta", "vega"):
                return FeatureCategory.RAW
            return FeatureCategory.DERIVED
        return FeatureCategory.RAW
    return FeatureCategory.RAW


def infer_lifecycle(category: FeatureCategory, name: str, parity_kind: FeatureComputationKind) -> FeatureLifecycle:
    if category == FeatureCategory.LOOKBACK:
        return FeatureLifecycle.SLIDING_WINDOW
    if category in (FeatureCategory.ROLLING, FeatureCategory.CUMULATIVE):
        return FeatureLifecycle.SESSION
    if category == FeatureCategory.TARGET:
        return FeatureLifecycle.TICK
    if category == FeatureCategory.METADATA:
        return FeatureLifecycle.TICK
    return FeatureLifecycle.TICK


def build_feature_policy_metadata(
    name: str,
    col: dict[str, Any] | None = None,
    *,
    group_id: str = "",
) -> FeaturePolicyMetadata:
    col = col or {}
    col_type = str(col.get("type") or "feature")
    gid = str(col.get("group") or group_id or "")
    parity = build_feature_parity_spec(name, gid)
    kind = parity.kind

    deps = list(col.get("depends_on") or infer_depends_on(name, gid, col_type))
    category = infer_feature_category(
        name, col_type=col_type, group_id=gid, parity_kind=kind, depends_on=deps,
    )
    lifecycle = infer_lifecycle(category, name, kind)

    samples, sec, mode = parse_warmup_string(parity.warmup)
    if name in _EMA_RATIO_WARMUP:
        samples = _EMA_RATIO_WARMUP[name]
    elif name in _WEIGHTED_EMA_WARMUP:
        samples = _WEIGHTED_EMA_WARMUP[name]

    policy_block = col.get("policy") or {}
    if policy_block.get("feature_category"):
        category = FeatureCategory(str(policy_block["feature_category"]))
    if policy_block.get("lifecycle"):
        lifecycle = FeatureLifecycle(str(policy_block["lifecycle"]))
    if policy_block.get("intrinsic_warmup_samples") is not None:
        samples = int(policy_block["intrinsic_warmup_samples"])
    if policy_block.get("intrinsic_warmup_sec") is not None:
        sec = int(policy_block["intrinsic_warmup_sec"])

    rolling_type = _infer_rolling_type(name) if category == FeatureCategory.ROLLING else None
    gap_sensitive = category == FeatureCategory.ROLLING
    reset_on_gap = category == FeatureCategory.ROLLING and lifecycle == FeatureLifecycle.SESSION

    if policy_block.get("gap_sensitive") is not None:
        gap_sensitive = bool(policy_block["gap_sensitive"])
    if policy_block.get("reset_on_gap") is not None:
        reset_on_gap = bool(policy_block["reset_on_gap"])

    anchor = _policy_anchor_name(name)
    policy_deps = _augment_dependencies(name, deps, category)

    formula_version = str(col.get("introduced_version") or col.get("formula_version") or "1")
    policy_version = str(policy_block.get("policy_version") or col.get("policy_version") or "1")

    meta = FeaturePolicyMetadata(
        name=name,
        feature_category=category,
        lifecycle=lifecycle,
        dependencies=policy_deps,
        intrinsic_warmup_samples=samples,
        intrinsic_warmup_sec=sec,
        warmup_mode=mode,
        rolling_type=rolling_type,
        gap_sensitive=gap_sensitive,
        reset_on_gap=reset_on_gap,
        formula_version=formula_version,
        policy_version=policy_version,
        group_id=gid,
        col_type=col_type,
        parity_kind=kind.value,
        policy_anchor=anchor,
    )
    return meta


def resolve_effective_warmup(
    meta: FeaturePolicyMetadata,
    registry: dict[str, FeaturePolicyMetadata],
) -> FeaturePolicyMetadata:
    """Compute effective warm-up (max over dependencies). Returns new metadata with fields set."""
    if meta.intrinsic_warmup_samples > 0 or meta.intrinsic_warmup_sec > 0:
        eff_samples = meta.intrinsic_warmup_samples
        eff_sec = meta.intrinsic_warmup_sec
        inherited = False
    else:
        eff_samples = 0
        eff_sec = 0
        inherited = meta.feature_category == FeatureCategory.DERIVED

    for dep in meta.dependencies:
        dep_meta = registry.get(dep)
        if not dep_meta:
            if dep.startswith("__roll."):
                parts = dep.split(".")
                if len(parts) >= 3 and parts[2].startswith("ema"):
                    try:
                        eff_samples = max(eff_samples, int(parts[2].replace("ema", "")))
                    except ValueError:
                        pass
            continue
        dep_eff = dep_meta.effective_warmup_samples or dep_meta.intrinsic_warmup_samples
        dep_sec = dep_meta.intrinsic_warmup_sec
        eff_samples = max(eff_samples, dep_eff)
        eff_sec = max(eff_sec, dep_sec)

    if meta.feature_category == FeatureCategory.LOOKBACK and meta.intrinsic_warmup_sec > 0:
        inherited = False
    elif meta.policy_anchor and meta.feature_category == FeatureCategory.DERIVED:
        inherited = True

    return FeaturePolicyMetadata(
        name=meta.name,
        feature_category=meta.feature_category,
        lifecycle=meta.lifecycle,
        dependencies=meta.dependencies,
        intrinsic_warmup_samples=meta.intrinsic_warmup_samples,
        intrinsic_warmup_sec=meta.intrinsic_warmup_sec,
        warmup_mode=meta.warmup_mode,
        rolling_type=meta.rolling_type,
        gap_sensitive=meta.gap_sensitive,
        reset_on_gap=meta.reset_on_gap,
        formula_version=meta.formula_version,
        policy_version=meta.policy_version,
        group_id=meta.group_id,
        col_type=meta.col_type,
        parity_kind=meta.parity_kind,
        effective_warmup_samples=eff_samples,
        effective_warmup_inherited=(
            inherited
            and (
                meta.policy_anchor is not None
                or (meta.intrinsic_warmup_samples == 0 and meta.intrinsic_warmup_sec == 0)
            )
        ),
        policy_anchor=meta.policy_anchor,
    )
