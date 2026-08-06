"""Feature grid parity — training, replay, and live must share one grid definition.

Golden rule: ``feature_grid_step_sec`` (from ``trainingIntervalSec`` / sampling) drives
all bar-based rollings. Calendar-named features use fixed wall-clock seconds in every mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from chain_replay_ml.ticks import EMA_BAR_INTERVAL_SEC

FEATURE_PARITY_VERSION = "1.2"

# Rolling z-score window in chain_maps.precompute_chain_maps (sample timestamps).
_CHAIN_STRADDLE_ZSCORE_BARS = 30

_RV_RULE_5M = (
    "Std dev of log returns over 300s window; "
    "returns sampled every feature_grid_step_sec"
)
_RV_RULE_10M = (
    "Std dev of log returns over 600s window; "
    "returns sampled every feature_grid_step_sec"
)
_RV_RATIO_RULE = "{num} / {den}; requires both RV windows populated"
_BODY_PREV1_RULE = (
    "OHLC body % of prior 10s candle (wall-clock ts−20s → ts−10s): "
    "(close − open) / open × 100"
)


def _ema_dep(name: str, base: str) -> str:
    m = _EMA_BAR_SUFFIX.search(name)
    period = m.group(1) if m else "?"
    return f"EMA{period}({base})"


def _rv_warmup_bars(lookback_sec: int, *, ref_grid_sec: int = 10) -> str:
    """Bar count at reference grid; scales with feature_grid_step_sec in live code."""
    bars = max(5, int(lookback_sec / max(ref_grid_sec, 1)))
    return f"{bars} bars"


class FeatureComputationKind(str, Enum):
    """How a feature resolves time."""

    GRID_BAR = "grid_bar"
    """N bars spaced ``feature_grid_step_sec`` apart (EMA, std-on-grid, sample rollings)."""

    CALENDAR_SEC = "calendar_sec"
    """Fixed wall-clock lookback; name encodes seconds/minutes (``spot_change_5s``)."""

    STATIC = "static"
    """Point-in-time from current ticks / session metadata — no rolling history."""


class FeatureSharedScope(str, Enum):
    """Where the engine may share computed values across rows."""

    CHAIN = "chain"
    TOKEN = "token"
    SPOT = "spot"
    SESSION = "session"


@dataclass(frozen=True)
class FeatureParitySpec:
    feature: str
    kind: FeatureComputationKind
    rule: str
    warmup: str
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    scope: FeatureSharedScope = FeatureSharedScope.TOKEN

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "kind": self.kind.value,
            "rule": self.rule,
            "warmup": self.warmup,
            "depends_on": list(self.depends_on),
            "scope": self.scope.value,
        }


# Explicit overrides — authoritative for listed features.
_EXPLICIT: dict[str, FeatureParitySpec] = {
    "spot": FeatureParitySpec(
        feature="spot",
        kind=FeatureComputationKind.STATIC,
        rule="Index LTP at sample timestamp (shared across all strikes)",
        warmup="0 seconds",
        depends_on=("spot",),
        scope=FeatureSharedScope.SPOT,
    ),
    "ltp_ema9_to_ltp_ratio": FeatureParitySpec(
        feature="ltp_ema9_to_ltp_ratio",
        kind=FeatureComputationKind.GRID_BAR,
        rule="EMA9(ltp) / ltp; 9 bars × feature_grid_step_sec",
        warmup="9 bars",
        depends_on=("ltp", "EMA9(ltp)", "feature_grid_step_sec"),
        scope=FeatureSharedScope.TOKEN,
    ),
    "ltp_ema20_to_ltp_ratio": FeatureParitySpec(
        feature="ltp_ema20_to_ltp_ratio",
        kind=FeatureComputationKind.GRID_BAR,
        rule="EMA20(ltp) / ltp; 20 bars × feature_grid_step_sec",
        warmup="20 bars",
        depends_on=("ltp", "EMA20(ltp)", "feature_grid_step_sec"),
        scope=FeatureSharedScope.TOKEN,
    ),
    "ltp_ema50_to_ltp_ratio": FeatureParitySpec(
        feature="ltp_ema50_to_ltp_ratio",
        kind=FeatureComputationKind.GRID_BAR,
        rule="EMA50(ltp) / ltp; 50 bars × feature_grid_step_sec",
        warmup="50 bars",
        depends_on=("ltp", "EMA50(ltp)", "feature_grid_step_sec"),
        scope=FeatureSharedScope.TOKEN,
    ),
    "ltp_ema100_to_ltp_ratio": FeatureParitySpec(
        feature="ltp_ema100_to_ltp_ratio",
        kind=FeatureComputationKind.GRID_BAR,
        rule="EMA100(ltp) / ltp; 100 bars × feature_grid_step_sec",
        warmup="100 bars",
        depends_on=("ltp", "EMA100(ltp)", "feature_grid_step_sec"),
        scope=FeatureSharedScope.TOKEN,
    ),
    "ltp_ema200_to_ltp_ratio": FeatureParitySpec(
        feature="ltp_ema200_to_ltp_ratio",
        kind=FeatureComputationKind.GRID_BAR,
        rule="EMA200(ltp) / ltp; 200 bars × feature_grid_step_sec",
        warmup="200 bars",
        depends_on=("ltp", "EMA200(ltp)", "feature_grid_step_sec"),
        scope=FeatureSharedScope.TOKEN,
    ),
    "ltp_std20_to_ltp_ratio": FeatureParitySpec(
        feature="ltp_std20_to_ltp_ratio",
        kind=FeatureComputationKind.GRID_BAR,
        rule="StdDev20(ltp on grid) / ltp; 20 bars × feature_grid_step_sec",
        warmup="20 bars",
        depends_on=("ltp", "feature_grid_step_sec"),
        scope=FeatureSharedScope.TOKEN,
    ),
    "spot_ema20_to_ltp_ratio": FeatureParitySpec(
        feature="spot_ema20_to_ltp_ratio",
        kind=FeatureComputationKind.GRID_BAR,
        rule="EMA20(spot) / ltp; spot EMA on feature grid",
        warmup="20 bars",
        depends_on=("spot", "EMA20(spot)", "ltp", "feature_grid_step_sec"),
        scope=FeatureSharedScope.TOKEN,
    ),
    "spot_change_5s": FeatureParitySpec(
        feature="spot_change_5s",
        kind=FeatureComputationKind.CALENDAR_SEC,
        rule="Spot % change over 5 wall-clock seconds",
        warmup="5 seconds",
        depends_on=("spot",),
        scope=FeatureSharedScope.SPOT,
    ),
    "body_pct_10s": FeatureParitySpec(
        feature="body_pct_10s",
        kind=FeatureComputationKind.CALENDAR_SEC,
        rule="OHLC body % over prior 10 wall-clock seconds on option LTP",
        warmup="10 seconds",
        depends_on=("ltp",),
        scope=FeatureSharedScope.TOKEN,
    ),
    "spot_rv_5m": FeatureParitySpec(
        feature="spot_rv_5m",
        kind=FeatureComputationKind.GRID_BAR,
        rule=_RV_RULE_5M,
        warmup=_rv_warmup_bars(300),
        depends_on=("spot", "feature_grid_step_sec"),
        scope=FeatureSharedScope.SPOT,
    ),
    "spot_rv_10m": FeatureParitySpec(
        feature="spot_rv_10m",
        kind=FeatureComputationKind.GRID_BAR,
        rule=_RV_RULE_10M,
        warmup=_rv_warmup_bars(600),
        depends_on=("spot", "feature_grid_step_sec"),
        scope=FeatureSharedScope.SPOT,
    ),
    "spot_rv_ratio": FeatureParitySpec(
        feature="spot_rv_ratio",
        kind=FeatureComputationKind.STATIC,
        rule=_RV_RATIO_RULE.format(num="spot_rv_5m", den="spot_rv_10m"),
        warmup="0 seconds",
        depends_on=("spot_rv_5m", "spot_rv_10m"),
        scope=FeatureSharedScope.SPOT,
    ),
    "opt_rv_5m": FeatureParitySpec(
        feature="opt_rv_5m",
        kind=FeatureComputationKind.GRID_BAR,
        rule=_RV_RULE_5M.replace("log returns", "option LTP log returns"),
        warmup=_rv_warmup_bars(300),
        depends_on=("ltp", "feature_grid_step_sec"),
        scope=FeatureSharedScope.TOKEN,
    ),
    "opt_rv_10m": FeatureParitySpec(
        feature="opt_rv_10m",
        kind=FeatureComputationKind.GRID_BAR,
        rule=_RV_RULE_10M.replace("log returns", "option LTP log returns"),
        warmup=_rv_warmup_bars(600),
        depends_on=("ltp", "feature_grid_step_sec"),
        scope=FeatureSharedScope.TOKEN,
    ),
    "opt_rv_ratio": FeatureParitySpec(
        feature="opt_rv_ratio",
        kind=FeatureComputationKind.STATIC,
        rule=_RV_RATIO_RULE.format(num="opt_rv_5m", den="opt_rv_10m"),
        warmup="0 seconds",
        depends_on=("opt_rv_5m", "opt_rv_10m"),
        scope=FeatureSharedScope.TOKEN,
    ),
    "spot_body_pct_prev1": FeatureParitySpec(
        feature="spot_body_pct_prev1",
        kind=FeatureComputationKind.CALENDAR_SEC,
        rule=_BODY_PREV1_RULE.replace("OHLC body", "Spot OHLC body"),
        warmup="20 seconds",
        depends_on=("spot",),
        scope=FeatureSharedScope.SPOT,
    ),
    "opt_body_pct_prev1": FeatureParitySpec(
        feature="opt_body_pct_prev1",
        kind=FeatureComputationKind.CALENDAR_SEC,
        rule=_BODY_PREV1_RULE.replace("OHLC body", "Option OHLC body"),
        warmup="20 seconds",
        depends_on=("ltp",),
        scope=FeatureSharedScope.TOKEN,
    ),
    "atm_straddle_zscore_30m": FeatureParitySpec(
        feature="atm_straddle_zscore_30m",
        kind=FeatureComputationKind.GRID_BAR,
        rule=f"Z-score over last {_CHAIN_STRADDLE_ZSCORE_BARS} sample bars (× feature_grid_step_sec)",
        warmup=f"{_CHAIN_STRADDLE_ZSCORE_BARS} bars",
        depends_on=("atm_straddle", "feature_grid_step_sec"),
        scope=FeatureSharedScope.CHAIN,
    ),
    "atm_straddle_zscore_change_5m": FeatureParitySpec(
        feature="atm_straddle_zscore_change_5m",
        kind=FeatureComputationKind.CALENDAR_SEC,
        rule="Z-score delta over 300 wall-clock seconds",
        warmup="300 seconds",
        depends_on=("atm_straddle_zscore_30m",),
        scope=FeatureSharedScope.CHAIN,
    ),
    "chain_pcr": FeatureParitySpec(
        feature="chain_pcr",
        kind=FeatureComputationKind.STATIC,
        rule="Chain-wide put/call OI ratio at sample timestamp",
        warmup="0 seconds",
        depends_on=("chain OI aggregates",),
        scope=FeatureSharedScope.CHAIN,
    ),
    "chain_pcr_change_5m": FeatureParitySpec(
        feature="chain_pcr_change_5m",
        kind=FeatureComputationKind.CALENDAR_SEC,
        rule="chain_pcr change over 300 wall-clock seconds",
        warmup="300 seconds",
        depends_on=("chain_pcr",),
        scope=FeatureSharedScope.CHAIN,
    ),
    "delta": FeatureParitySpec(
        feature="delta",
        kind=FeatureComputationKind.STATIC,
        rule="Black–Scholes delta at sample timestamp",
        warmup="0 seconds",
        depends_on=("ltp", "spot", "strike", "current_iv", "minutes_to_expiry"),
        scope=FeatureSharedScope.TOKEN,
    ),
    "dgt_reiv_pred_change_10s": FeatureParitySpec(
        feature="dgt_reiv_pred_change_10s",
        kind=FeatureComputationKind.CALENDAR_SEC,
        rule="DGT REIV delta over 10 wall-clock seconds",
        warmup="10 seconds",
        depends_on=("dgt_reiv_pred",),
        scope=FeatureSharedScope.TOKEN,
    ),
    "delta_change_5m": FeatureParitySpec(
        feature="delta_change_5m",
        kind=FeatureComputationKind.CALENDAR_SEC,
        rule="Delta change over 300 wall-clock seconds",
        warmup="300 seconds",
        depends_on=("delta",),
        scope=FeatureSharedScope.TOKEN,
    ),
}

# group_id → default scope for inference
_GROUP_SCOPE: dict[str, FeatureSharedScope] = {
    "chain": FeatureSharedScope.CHAIN,
    "atm_straddle": FeatureSharedScope.CHAIN,
    "atm6_ltp": FeatureSharedScope.CHAIN,
    "time": FeatureSharedScope.SESSION,
    "price": FeatureSharedScope.TOKEN,
    "greeks": FeatureSharedScope.TOKEN,
    "iv": FeatureSharedScope.TOKEN,
    "oi": FeatureSharedScope.TOKEN,
    "volume": FeatureSharedScope.TOKEN,
    "momentum": FeatureSharedScope.SPOT,
    "sharp_momentum": FeatureSharedScope.SPOT,
    "iv_zscore": FeatureSharedScope.TOKEN,
    "iv_ema_ratio": FeatureSharedScope.TOKEN,
    "spot_hl": FeatureSharedScope.SPOT,
    "advanced": FeatureSharedScope.TOKEN,
    "historical": FeatureSharedScope.TOKEN,
}

_SEC_SUFFIX = re.compile(r"_(\d+)s(?:_|$)")
_MIN_SUFFIX = re.compile(r"_(\d+)m(?:_|$)")
_EMA_BAR_SUFFIX = re.compile(r"(?:ltp_ema|spot_ema|iv_ema|ema)(\d+)", re.I)
_STD_BARS = re.compile(r"(?:ltp_std|std)(\d+)", re.I)


def resolve_feature_grid_step_sec(
    *,
    ctx: Any | None = None,
    sampling: dict[str, Any] | None = None,
    dataset_configuration: dict[str, Any] | None = None,
    fallback: int | None = None,
) -> int:
    """Single resolver used by dataset build, replay inference, and live engine."""
    if ctx is not None:
        step = getattr(ctx, "feature_grid_step_sec", None)
        if step is not None and int(step) > 0:
            return max(int(step), 1)
    cfg = dataset_configuration or {}
    for key in ("feature_grid_step_sec", "sampling_interval_sec"):
        raw = cfg.get(key)
        if raw is not None and int(raw) > 0:
            return max(int(raw), 1)
    samp = sampling or {}
    raw = samp.get("trainingIntervalSec") or samp.get("interval_sec")
    if raw is not None and int(raw) > 0:
        return max(int(raw), 1)
    if fallback is not None and int(fallback) > 0:
        return max(int(fallback), 1)
    return max(int(EMA_BAR_INTERVAL_SEC), 1)


def rv_subsample_step_sec(grid_step_sec: int | float) -> float:
    """Spacing between return points inside realized-vol windows."""
    return max(float(grid_step_sec), 1.0)


def _calendar_sec_from_name(name: str) -> int | None:
    n = str(name)
    m = _SEC_SUFFIX.search(n)
    if m:
        return int(m.group(1))
    m = _MIN_SUFFIX.search(n)
    if m:
        return int(m.group(1)) * 60
    if "_1m" in n and "_10m" not in n and "_15m" not in n:
        return 60
    if "_5m" in n:
        return 300
    if "_10m" in n:
        return 600
    if "_15m" in n:
        return 900
    return None


def _grid_bars_from_name(name: str) -> int | None:
    n = str(name)
    m = _EMA_BAR_SUFFIX.search(n)
    if m:
        return int(m.group(1))
    m = _STD_BARS.search(n)
    if m:
        return int(m.group(1))
    if n == "atm_straddle_zscore_30m":
        return _CHAIN_STRADDLE_ZSCORE_BARS
    return None


def infer_shared_scope(name: str, group_id: str = "") -> FeatureSharedScope:
    n = str(name)
    if n in _EXPLICIT:
        return _EXPLICIT[n].scope
    if n == "spot" or n in ("spot_ltp", "index_ltp", "underlying_ltp"):
        return FeatureSharedScope.SPOT
    if group_id and group_id in _GROUP_SCOPE:
        base = _GROUP_SCOPE[group_id]
        if base == FeatureSharedScope.SPOT and n.startswith(("ltp_", "opt_", "oi_", "delta", "gamma", "theta", "vega")):
            return FeatureSharedScope.TOKEN
        return base
    if n.startswith(("chain_", "atm_straddle", "atm_pcr", "max_call_oi", "max_put_oi", "ce_atm6", "pe_atm6")):
        return FeatureSharedScope.CHAIN
    if n.startswith("spot_") or n.startswith(("spot_change", "spot_return", "spot_rv", "spot_vol", "spot_ema", "spot_dist", "spot_body")):
        return FeatureSharedScope.SPOT
    if n in ("minutes_to_expiry", "minutes_since_open", "minutes_to_close", "is_first_hour", "is_last_hour", "minute_of_day"):
        return FeatureSharedScope.SESSION
    return FeatureSharedScope.TOKEN


def classify_feature(name: str, group_id: str = "") -> FeatureComputationKind:
    """Best-effort kind for audit / investigation UI."""
    if name in _EXPLICIT:
        return _EXPLICIT[name].kind
    n = str(name)
    # Historic multi-TF EMAs are as-of lookups of a precomputed bar series
    # (warmup already applied in angel_historic_bars). Do NOT treat as GRID_BAR
    # — that incorrectly applies N sample warm-up on the option tick grid.
    if group_id == "historic_spot_ema" or re.match(
        r"^spot_(?:1m|3m|5m|15m)_ema\d+$", n, re.I
    ):
        return FeatureComputationKind.STATIC
    if _grid_bars_from_name(n) is not None:
        return FeatureComputationKind.GRID_BAR
    if n.startswith(("spot_rv_", "opt_rv_")) and not n.endswith("_ratio"):
        return FeatureComputationKind.GRID_BAR
    if n.endswith("_rv_ratio") or n in ("spot_rv_ratio", "opt_rv_ratio", "rv_ratio"):
        return FeatureComputationKind.STATIC
    cal = _calendar_sec_from_name(n)
    if cal is not None:
        return FeatureComputationKind.CALENDAR_SEC
    if any(tag in n for tag in ("_change_", "_return_", "_flow_", "body_pct_", "range_pct_", "vol_ratio_")):
        return FeatureComputationKind.CALENDAR_SEC
    if group_id in ("time",):
        return FeatureComputationKind.STATIC
    if group_id in ("greeks", "moneyness", "price") and not _calendar_sec_from_name(n):
        if n.startswith(("ltp_return", "spot_change", "oi_change", "volume_change")):
            return FeatureComputationKind.CALENDAR_SEC
        return FeatureComputationKind.STATIC
    return FeatureComputationKind.STATIC


def infer_warmup(name: str, kind: FeatureComputationKind) -> str:
    if name in _EXPLICIT:
        return _EXPLICIT[name].warmup
    bars = _grid_bars_from_name(name)
    if kind == FeatureComputationKind.GRID_BAR and bars is not None:
        return f"{bars} bars"
    if kind == FeatureComputationKind.GRID_BAR and name.startswith(("spot_rv_", "opt_rv_")):
        if "_5m" in name:
            return _rv_warmup_bars(300)
        if "_10m" in name:
            return _rv_warmup_bars(600)
        return _rv_warmup_bars(300)
    if kind == FeatureComputationKind.GRID_BAR:
        return "1 bars"
    cal = _calendar_sec_from_name(name)
    if kind == FeatureComputationKind.CALENDAR_SEC and cal is not None:
        if "body_pct_prev1" in name or "range_pct_prev1" in name:
            return f"{cal + 10} seconds"
        return f"{cal} seconds"
    if kind == FeatureComputationKind.STATIC:
        return "0 seconds"
    return "0 seconds"


def infer_rule(name: str, kind: FeatureComputationKind) -> str:
    if name in _EXPLICIT:
        return _EXPLICIT[name].rule
    n = str(name)
    bars = _grid_bars_from_name(n)
    if kind == FeatureComputationKind.GRID_BAR and bars is not None:
        base = "spot" if "spot_ema" in n else "ltp"
        return f"EMA{bars}({base}) / {base}; {bars} bars × feature_grid_step_sec" if "ema" in n.lower() and "ratio" in n else f"{bars} bars × feature_grid_step_sec"
    if kind == FeatureComputationKind.GRID_BAR and n.startswith("spot_rv_"):
        if "_5m" in n:
            return _RV_RULE_5M
        if "_10m" in n:
            return _RV_RULE_10M
    if kind == FeatureComputationKind.GRID_BAR and n.startswith("opt_rv_"):
        if "_5m" in n:
            return _RV_RULE_5M.replace("log returns", "option LTP log returns")
        if "_10m" in n:
            return _RV_RULE_10M.replace("log returns", "option LTP log returns")
    if n.endswith("_rv_ratio") or n == "rv_ratio":
        prefix = "spot" if n.startswith("spot") else "opt"
        return _RV_RATIO_RULE.format(num=f"{prefix}_rv_5m", den=f"{prefix}_rv_10m")
    if "body_pct_prev1" in n:
        label = "Spot" if n.startswith("spot_") else "Option"
        return _BODY_PREV1_RULE.replace("OHLC body", f"{label} OHLC body")
    cal = _calendar_sec_from_name(n)
    if kind == FeatureComputationKind.CALENDAR_SEC and cal is not None:
        qty = "spot" if n.startswith("spot_") or "spot" in n else "ltp"
        return f"{qty} change or OHLC metric over {cal} wall-clock seconds"
    if kind == FeatureComputationKind.STATIC:
        if n.endswith("_ratio") and "_rv_" in n:
            return "Ratio of two computed features at sample timestamp"
        return "Point-in-time value at sample timestamp"
    return "Point-in-time value at sample timestamp"


def infer_depends_on(name: str, kind: FeatureComputationKind, scope: FeatureSharedScope) -> tuple[str, ...]:
    if name in _EXPLICIT:
        return _EXPLICIT[name].depends_on
    deps: list[str] = []
    n = str(name)
    if kind == FeatureComputationKind.GRID_BAR:
        deps.append("feature_grid_step_sec")
        if n.startswith("spot_ema") and "ratio" in n:
            deps.extend(["spot", _ema_dep(n, "spot"), "ltp"])
        elif n.startswith("spot_ema"):
            deps.extend(["spot", _ema_dep(n, "spot")])
        elif "ema" in n.lower() and "ratio" in n:
            deps.extend(["ltp", _ema_dep(n, "ltp")])
        elif n.startswith(("spot_rv_",)):
            deps.append("spot")
        elif n.startswith(("opt_rv_",)):
            deps.append("ltp")
        elif scope == FeatureSharedScope.SPOT or n.startswith("spot_"):
            deps.append("spot")
        else:
            deps.append("ltp")
    elif kind == FeatureComputationKind.CALENDAR_SEC:
        if n.startswith("spot_") or n.startswith("chain_") and "pcr" in n:
            if "pcr" in n and "_change" in n:
                base = n.rsplit("_change", 1)[0]
                deps.append(base)
            elif "spot" in n or n.startswith("spot_"):
                deps.append("spot")
            else:
                deps.append("chain aggregates")
        elif n.startswith("chain_"):
            deps.append("chain aggregates")
        elif "_change_" in n:
            base = re.sub(r"_change_\d+[sm]", "", n)
            if base and base != n:
                deps.append(base)
            else:
                deps.append("ltp")
        else:
            deps.append("ltp" if scope == FeatureSharedScope.TOKEN else "spot")
    else:
        if n.endswith("_rv_ratio"):
            deps.extend([n.replace("_ratio", "_5m"), n.replace("_ratio", "_10m")])
        elif scope == FeatureSharedScope.CHAIN:
            deps.append("chain aggregates")
        elif scope == FeatureSharedScope.SPOT:
            deps.append("spot")
        elif scope == FeatureSharedScope.SESSION:
            pass
        else:
            deps.append("ltp")
    return tuple(dict.fromkeys(deps))


def build_feature_parity_spec(name: str, group_id: str = "") -> FeatureParitySpec:
    """Build full parity row for one registry feature."""
    if name in _EXPLICIT:
        return _EXPLICIT[name]
    kind = classify_feature(name, group_id)
    scope = infer_shared_scope(name, group_id)
    return FeatureParitySpec(
        feature=name,
        kind=kind,
        rule=infer_rule(name, kind),
        warmup=infer_warmup(name, kind),
        depends_on=infer_depends_on(name, kind, scope),
        scope=scope,
    )


def feature_parity_for_name(name: str, group_id: str = "") -> dict[str, Any]:
    return build_feature_parity_spec(name, group_id).as_dict()


def max_warmup_bars(specs: list[FeatureParitySpec]) -> int:
    """Longest grid-bar warmup across a feature set (for live session hints)."""
    best = 0
    for spec in specs:
        if spec.kind != FeatureComputationKind.GRID_BAR:
            continue
        m = re.match(r"^(\d+)\s+bars?$", spec.warmup.strip(), re.I)
        if m:
            best = max(best, int(m.group(1)))
    return best


def validate_warmup_units(spec: FeatureParitySpec) -> bool:
    """True when warmup string matches kind convention (bars / seconds / 0 seconds)."""
    w = spec.warmup.strip().lower()
    if spec.kind == FeatureComputationKind.GRID_BAR:
        return bool(re.match(r"^\d+\s+bars?$", w))
    if spec.kind == FeatureComputationKind.CALENDAR_SEC:
        return bool(re.match(r"^\d+\s+seconds?$", w))
    if spec.kind == FeatureComputationKind.STATIC:
        return w == "0 seconds"
    return False


def feature_parity_audit_rows(*, include_all_registry: bool = True) -> list[dict[str, Any]]:
    """Full catalog for tooling — explicit entries plus inferred registry features."""
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []

    for name in sorted(_EXPLICIT.keys()):
        rows.append(_EXPLICIT[name].as_dict())
        seen.add(name)

    if include_all_registry:
        try:
            from .schema_registry import load_schema_registry

            reg = load_schema_registry()
            groups_meta = reg.get("groups") or {}
            feat_to_gid: dict[str, str] = {}
            for gid, block in groups_meta.items():
                for feat in (block or {}).get("features") or []:
                    feat_to_gid[str(feat)] = str(gid)
            cols = reg.get("columns") or {}
            for name, col in cols.items():
                if str(col.get("type") or "").lower() != "feature":
                    continue
                if name in seen:
                    continue
                gid = feat_to_gid.get(name, str(col.get("group") or ""))
                rows.append(build_feature_parity_spec(name, gid).as_dict())
                seen.add(name)
        except Exception:
            pass

    rows.sort(key=lambda r: str(r.get("feature") or ""))
    return rows


# Backward-compatible alias
def classify_feature_kind(name: str) -> str:
    return classify_feature(name).value
