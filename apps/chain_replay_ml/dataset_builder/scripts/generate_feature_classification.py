#!/usr/bin/env python3
"""Generate FEATURE_CLASSIFICATION.md — 240-feature controller / lookback / stateless sheet."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_SCRIPT_DIR)
from path_config import ensure_ml_studio_paths

ensure_ml_studio_paths()

from chain_replay_ml.dataset_builder.feature_plugins import (  # noqa: E402
    _REGISTRY_FEATURES,
    horizon_label,
)
from chain_replay_ml.feature_policy.registry import load_feature_policy_registry  # noqa: E402
from chain_replay_ml.feature_policy.types import FeatureCategory  # noqa: E402

from chain_replay_ml.dataset_builder.classification_validate import (  # noqa: E402
    generated_document_header,
    render_dependency_graph_markdown,
    validate_governance,
)
from chain_replay_ml.dataset_builder.controller_registry import (  # noqa: E402
    CONTROLLER_FEATURES,
    CONTROLLER_REGISTRY,
)

_REPO_ROOT = os.path.abspath(os.path.join(_PKG_DIR, "..", "..", "..", ".."))
_DOCS_CONTROLLERS = os.path.join(_REPO_ROOT, "docs", "controllers")
OUT_PATH = os.path.join(_DOCS_CONTROLLERS, "FEATURE_CLASSIFICATION.md")
DEP_GRAPH_PATH = os.path.join(_DOCS_CONTROLLERS, "CONTROLLER_DEPENDENCY_GRAPH.md")

SPOT_RESET_ON_TOKEN_GAP = "No"

TOKEN_CONTROLLERS = frozenset(
    k for k in CONTROLLER_FEATURES if k.startswith("token.") or k == "composite.weighted_ltp_ema"
)
SPOT_CONTROLLERS = frozenset(
    k for k in CONTROLLER_FEATURES
    if k.startswith("spot.") or k in (
        "composite.weighted_spot_ema", "composite.weighted_spot_hl", "composite.iv_x_spot_ema",
    )
)
DERIVED_READERS = frozenset({"token.rv.ratio", "spot.rv.ratio"})

COMPOSITE_EMITTERS = frozenset({
    "composite.weighted_ltp_ema",
    "composite.weighted_spot_ema",
    "composite.weighted_spot_hl",
    "composite.iv_x_spot_ema",
})

SOURCE_CONTROLLERS: dict[str, list[str]] = {
    cid: list(spec.source_controllers) for cid, spec in CONTROLLER_REGISTRY.items() if spec.source_controllers
}
SOURCE_CONTROLLERS.setdefault("token.rv.ratio", ["token.rv.5m", "token.rv.10m"])
SOURCE_CONTROLLERS.setdefault("spot.rv.ratio", ["spot.rv.5m", "spot.rv.10m"])

STATE_READS: dict[str, str] = {
    "composite.weighted_ltp_ema": "Reads: ema9.value, ema20.value, ema50.value, ema200.value",
    "composite.weighted_spot_ema": "Reads: spot.ema9.value, spot.ema20.value, spot.ema50.value, spot.ema100.value, spot.ema200.value",
    "composite.weighted_spot_hl": "Reads: spot.hl.ema20…300 high/low .value bands",
    "composite.iv_x_spot_ema": "Reads: iv_window.*.value + spot.ema9…200.value",
    "token.rv.ratio": "Reads: opt_rv_5m.value, opt_rv_10m.value",
    "spot.rv.ratio": "Reads: spot_rv_5m.value, spot_rv_10m.value",
}

CONTROLLER_PHASE: dict[str, str] = {cid: str(spec.phase) for cid, spec in CONTROLLER_REGISTRY.items()}

CONTROLLER_STATE_STORAGE: dict[str, str] = {
    "token.ltp.ema9": "ema_value",
    "token.ltp.ema20": "ema_value",
    "token.ltp.ema50": "ema_value",
    "token.ltp.ema100": "ema_value",
    "token.ltp.ema200": "ema_value",
    "composite.weighted_ltp_ema": "Reads: ema9.value, ema20.value, ema50.value, ema200.value",
    "token.ltp.std20": "ring_buffer",
    "token.rv.5m": "return_buffer",
    "token.rv.10m": "return_buffer",
    "token.rv.ratio": "Reads: opt_rv_5m.value, opt_rv_10m.value",
    "token.iv_window.1m": "ring_buffer",
    "token.iv_window.5m": "ring_buffer",
    "token.iv_window.15m": "ring_buffer",
    "token.iv_window.30m": "ring_buffer",
    "token.iv_window.session": "session_deque",
    "token.iv_history.1m": "deque",
    "token.iv_history.5m": "deque",
    "token.iv_history.15m": "deque",
    "token.roll": "snapshot",
    "token.dgt": "snapshot_history",
    "spot.ema9": "ema_value",
    "spot.ema20": "ema_value",
    "spot.ema50": "ema_value",
    "spot.ema100": "ema_value",
    "spot.ema200": "ema_value",
    "spot.ema300": "ema_value",
    "spot.hl.ema20": "ema_value (high/low)",
    "spot.hl.ema50": "ema_value (high/low)",
    "spot.hl.ema100": "ema_value (high/low)",
    "spot.hl.ema200": "ema_value (high/low)",
    "spot.hl.ema300": "ema_value (high/low)",
    "spot.rv.5m": "return_buffer",
    "spot.rv.10m": "return_buffer",
    "spot.rv.ratio": "Reads: spot_rv_5m.value, spot_rv_10m.value",
    "spot.momentum": "crossover_state",
    "composite.weighted_spot_ema": "Reads: spot.ema9.value, spot.ema20.value, spot.ema50.value, spot.ema100.value, spot.ema200.value",
    "composite.weighted_spot_hl": "Reads: spot.hl.ema20…300 high/low .value bands",
    "composite.iv_x_spot_ema": "Reads: iv_window.*.value + spot.ema9…200.value",
}

CONTROLLER_WARMUP: dict[str, tuple[str, str]] = {
    cid: (spec.warmup_type, spec.warmup_value) for cid, spec in CONTROLLER_REGISTRY.items()
}

FEATURE_DEPENDS: dict[str, str] = {
    "ltp_ema9_to_ltp_ratio": "token.ltp.ema9 + LTP",
    "ltp_ema20_to_ltp_ratio": "token.ltp.ema20 + LTP",
    "ltp_ema50_to_ltp_ratio": "token.ltp.ema50 + LTP",
    "ltp_ema100_to_ltp_ratio": "token.ltp.ema100 + LTP",
    "ltp_ema200_to_ltp_ratio": "token.ltp.ema200 + LTP",
    "ltp_ema9_to_spot_ratio": "token.ltp.ema9 + LTP + spot",
    "ltp_ema20_to_spot_ratio": "token.ltp.ema20 + LTP + spot",
    "ltp_ema50_to_spot_ratio": "token.ltp.ema50 + LTP + spot",
    "ltp_ema100_to_spot_ratio": "token.ltp.ema100 + LTP + spot",
    "ltp_ema200_to_spot_ratio": "token.ltp.ema200 + LTP + spot",
    "weighted_ltp_ema_to_ltp_ratio": "token.ltp.ema9 + token.ltp.ema20 + token.ltp.ema50 + token.ltp.ema200 + LTP",
    "weighted_spot_ema_to_ltp_ratio": "spot.ema9 + spot.ema20 + spot.ema50 + spot.ema100 + spot.ema200 + LTP",
    "weighted_spot_ema_to_ltp_ratio_x_delta": "spot.ema9 + spot.ema20 + spot.ema50 + spot.ema100 + spot.ema200 + LTP + delta",
    "weighted_spot_ema_to_ltp_ratio_x_moneyness": "spot.ema9 + spot.ema20 + spot.ema50 + spot.ema100 + spot.ema200 + LTP + moneyness",
    "weighted_spot_ema_to_ltp_ratio_x_moneyness_x_delta": "spot.ema9 + spot.ema20 + spot.ema50 + spot.ema100 + spot.ema200 + LTP + moneyness + delta",
    "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_1m": "spot.ema9 + spot.ema20 + spot.ema50 + spot.ema100 + spot.ema200 + LTP + iv_zscore_1m",
    "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_5m": "spot.ema9 + spot.ema20 + spot.ema50 + spot.ema100 + spot.ema200 + LTP + iv_zscore_5m",
    "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m": "spot.ema9 + spot.ema20 + spot.ema50 + spot.ema100 + spot.ema200 + LTP + iv_zscore_15m",
    "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_1m_x_delta": "spot.ema9 + spot.ema20 + spot.ema50 + spot.ema100 + spot.ema200 + LTP + iv_zscore_1m + delta",
    "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_5m_x_delta": "spot.ema9 + spot.ema20 + spot.ema50 + spot.ema100 + spot.ema200 + LTP + iv_zscore_5m + delta",
    "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m_x_delta": "spot.ema9 + spot.ema20 + spot.ema50 + spot.ema100 + spot.ema200 + LTP + iv_zscore_15m + delta",
    "weighted_iv_zscore_x_weighted_spot_ema_to_ltp_ratio": "token.iv_window.* + spot.ema9 + spot.ema20 + spot.ema50 + spot.ema100 + spot.ema200 + LTP + IV",
    "ltp_ema9_to_spot_ratio_x_iv_ema9": "token.ltp.ema9 + token.iv.ema9 + spot",
    "ltp_ema20_to_spot_ratio_x_iv_ema20": "token.ltp.ema20 + token.iv.ema20 + spot",
    "ltp_ema50_to_spot_ratio_x_iv_ema50": "token.ltp.ema50 + token.iv.ema50 + spot",
    "ltp_ema100_to_spot_ratio_x_iv_ema100": "token.ltp.ema100 + token.iv.ema100 + spot",
    "ltp_ema200_to_spot_ratio_x_iv_ema200": "token.ltp.ema200 + token.iv.ema200 + spot",
    "ltp_ema300_to_spot_ratio_x_iv_ema300": "token.ltp.ema300 + token.iv.ema300 + spot",
    "spot_to_ltp_ratio_x_iv_ema9": "token.iv.ema9 + spot + LTP",
    "spot_to_ltp_ratio_x_iv_ema20": "token.iv.ema20 + spot + LTP",
    "spot_to_ltp_ratio_x_iv_ema50": "token.iv.ema50 + spot + LTP",
    "spot_to_ltp_ratio_x_iv_ema100": "token.iv.ema100 + spot + LTP",
    "spot_to_ltp_ratio_x_iv_ema200": "token.iv.ema200 + spot + LTP",
    "spot_to_ltp_ratio_x_iv_ema300": "token.iv.ema300 + spot + LTP",
    "iv_ema9_to_ltp_ratio": "token.iv.ema9 + LTP",
    "iv_ema20_to_ltp_ratio": "token.iv.ema20 + LTP",
    "iv_ema50_to_ltp_ratio": "token.iv.ema50 + LTP",
    "iv_ema100_to_ltp_ratio": "token.iv.ema100 + LTP",
    "iv_ema200_to_ltp_ratio": "token.iv.ema200 + LTP",
    "iv_ema300_to_ltp_ratio": "token.iv.ema300 + LTP",
    "iv_ema9_to_spot_ratio": "token.iv.ema9 + spot",
    "iv_ema20_to_spot_ratio": "token.iv.ema20 + spot",
    "iv_ema50_to_spot_ratio": "token.iv.ema50 + spot",
    "iv_ema100_to_spot_ratio": "token.iv.ema100 + spot",
    "iv_ema200_to_spot_ratio": "token.iv.ema200 + spot",
    "iv_ema300_to_spot_ratio": "token.iv.ema300 + spot",
    "spot_to_ltp_ratio_x_iv_ema9_x_moneyness": "token.iv.ema9 + spot + LTP + moneyness",
    "spot_to_ltp_ratio_x_iv_ema20_x_moneyness": "token.iv.ema20 + spot + LTP + moneyness",
    "spot_to_ltp_ratio_x_iv_ema50_x_moneyness": "token.iv.ema50 + spot + LTP + moneyness",
    "spot_to_ltp_ratio_x_iv_ema100_x_moneyness": "token.iv.ema100 + spot + LTP + moneyness",
    "spot_to_ltp_ratio_x_iv_ema200_x_moneyness": "token.iv.ema200 + spot + LTP + moneyness",
    "spot_to_ltp_ratio_x_iv_ema300_x_moneyness": "token.iv.ema300 + spot + LTP + moneyness",
    "spot_ema300_to_ltp_ratio": "spot.ema300 + LTP",
    "spot_ema9_to_ltp_ratio_x_moneyness": "spot.ema9 + LTP + moneyness",
    "spot_ema20_to_ltp_ratio_x_moneyness": "spot.ema20 + LTP + moneyness",
    "spot_ema50_to_ltp_ratio_x_moneyness": "spot.ema50 + LTP + moneyness",
    "spot_ema100_to_ltp_ratio_x_moneyness": "spot.ema100 + LTP + moneyness",
    "spot_ema200_to_ltp_ratio_x_moneyness": "spot.ema200 + LTP + moneyness",
    "spot_ema300_to_ltp_ratio_x_moneyness": "spot.ema300 + LTP + moneyness",
    "ltp_ema9_to_spot_ratio_x_moneyness": "token.ltp.ema9 + spot + moneyness",
    "ltp_ema20_to_spot_ratio_x_moneyness": "token.ltp.ema20 + spot + moneyness",
    "ltp_ema50_to_spot_ratio_x_moneyness": "token.ltp.ema50 + spot + moneyness",
    "ltp_ema100_to_spot_ratio_x_moneyness": "token.ltp.ema100 + spot + moneyness",
    "ltp_ema200_to_spot_ratio_x_moneyness": "token.ltp.ema200 + spot + moneyness",
    "ltp_ema300_to_spot_ratio_x_moneyness": "token.ltp.ema300 + spot + moneyness",
    "opt_rv_ratio": "opt_rv_5m + opt_rv_10m",
    "spot_rv_ratio": "spot_rv_5m + spot_rv_10m",
}

FEATURE_TO_CONTROLLER: dict[str, str] = {}
for ctrl, feats in CONTROLLER_FEATURES.items():
    for feat in feats:
        FEATURE_TO_CONTROLLER[feat] = ctrl

MANUAL_JUDGMENTS: list[str] = [
    "Memory model **Lookback** = history-based row scan; **Gap policy = StopAtGap** (boundary enforced, not a controller reset).",
    "Spot controllers: **Gap policy = Continue** — per-token `row_gap_exceeds` does not reset the market-wide spot stream.",
    "Composite emitters: **Memory model = Controller-derived**; **Emission policy = ControllerDerived**; no own rolling state.",
    "atm_straddle_zscore_30m: Lookback today; Future owner ChainController (Phase 3).",
    "sharp_momentum (18 features): Stateless levels today; Future owner spot.momentum (Phase 3).",
    "token.dgt lag/error: snapshot_history; dgt_reiv_pred owned by token.roll.",
    "ltp_to_dgt_reiv_ratio, ltp_to_bs_reiv_ratio, bs_reiv_to_ltp_ratio: AlwaysValid point-in-time ratios.",
]

# Wave 4: Master emits canonical score/count levels (packaging → Interaction).
_SHARP_MOMENTUM = frozenset(
    f"spot_{side}_score_{h}"
    for h in ("1m", "3m", "5m", "10m")
    for side in ("up", "down")
) | frozenset(
    f"spot_{side}_sample_count_{h}"
    for h in ("1m", "3m", "5m", "10m")
    for side in ("up", "down")
)

_CHAIN_SNAPSHOT = frozenset(_REGISTRY_FEATURES.get("chain", [])) | frozenset(
    _REGISTRY_FEATURES.get("atm6_ltp", [])
) | frozenset(
    n for n in sum(_REGISTRY_FEATURES.values(), [])
    if any(x in n for x in ("atm6", "chain_pcr", "atm_pcr", "ce_pe_atm", "ce_atm6", "pe_atm6"))
)

_TIME_FEATURES = frozenset(_REGISTRY_FEATURES.get("time", []))
_RAW_TICK = frozenset({
    "spot", "ltp", "bid_ask_spread", "option_vwap", "futures_ltp", "futures_vwap",
    "futures_day_volume", "futures_oi", "futures_bid", "futures_ask", "futures_spread",
    "option_oi", "option_day_volume", "option_bid", "option_ask",
    "spot_open", "spot_high", "spot_low", "spot_prev_close",
    "option_open", "option_high", "option_low", "option_prev_close",
    "ltq", "total_buy_qty", "total_sell_qty",
    "strike", "is_call",
    "delta", "gamma", "theta", "vega", "vanna", "volga", "charm", "speed", "current_iv", "oi", "volume",
})
_CAL_RE = re.compile(r"_(\d+)(s|sec|m|min)\b", re.I)


@dataclass
class ClassificationRow:
    feature: str
    group_id: str
    phase: str
    controller_owner: str
    future_owner: str
    source_controllers: str
    memory_model: str
    gap_policy: str
    warmup_type: str
    warmup_value: str
    emission_policy: str
    depends_on: str
    state_storage: str
    judgment: str | None = None


def _ema_value_read(ctrl: str) -> str:
    if ctrl.startswith("token.ltp.ema"):
        p = ctrl.rsplit(".", 1)[-1].replace("ema", "")
        return f"ema{p}.value"
    if ctrl.startswith("spot.ema") and not ctrl.startswith("spot.hl."):
        p = ctrl.rsplit(".", 1)[-1].replace("ema", "")
        return f"spot.ema{p}.value"
    if ctrl.startswith("spot.hl."):
        p = ctrl.rsplit(".", 1)[-1].replace("ema", "")
        return f"spot.hl.high_ema{p}.value, spot.hl.low_ema{p}.value"
    return f"{ctrl}.value"


def _source_controllers_label(ctrl: str) -> str:
    if ctrl in SOURCE_CONTROLLERS:
        return ", ".join(SOURCE_CONTROLLERS[ctrl])
    if ctrl in COMPOSITE_EMITTERS or ctrl in DERIVED_READERS:
        return ", ".join(SOURCE_CONTROLLERS.get(ctrl, [ctrl]))
    return ctrl


def _state_storage_for(ctrl: str, memory_model: str) -> str:
    if ctrl in STATE_READS:
        return STATE_READS[ctrl]
    if memory_model == "Controller-derived":
        return STATE_READS.get(ctrl, f"Reads: {_ema_value_read(ctrl)}")
    if memory_model == "Controller":
        stored = CONTROLLER_STATE_STORAGE.get(ctrl)
        if stored:
            if stored == "ema_value":
                return f"Reads: {_ema_value_read(ctrl)}"
            return stored
        return f"Reads: {_ema_value_read(ctrl)}"
    return "—"


def _gap_policy_for(ctrl: str | None, memory_model: str) -> str:
    if memory_model == "Lookback":
        return "StopAtGap"
    if memory_model == "Stateless":
        return "None"
    if ctrl and ctrl in TOKEN_CONTROLLERS:
        return "Reset"
    if ctrl and ctrl in SPOT_CONTROLLERS:
        return "Continue"
    return "None"


def _feature_groups() -> dict[str, str]:
    out: dict[str, str] = {}
    for gid, feats in _REGISTRY_FEATURES.items():
        for f in feats:
            out[f] = gid
    return out


def _all_features() -> list[str]:
    feats: list[str] = []
    for _gid, names in _REGISTRY_FEATURES.items():
        feats.extend(names)
    return feats


def _lookback_horizon(name: str) -> str | None:
    m = _CAL_RE.search(name)
    if not m:
        return None
    val, unit = int(m.group(1)), m.group(2).lower()
    sec = val * 60 if unit.startswith("m") else val
    return horizon_label(sec)


def _is_chain_snapshot(name: str, group_id: str) -> bool:
    if name in ("atm_straddle",):
        return True
    if group_id in ("chain", "atm6_ltp") and "change" not in name and "slope" not in name and "zscore" not in name:
        return name in _CHAIN_SNAPSHOT or group_id in ("chain", "atm6_ltp")
    if name in _CHAIN_SNAPSHOT and "change" not in name and "slope" not in name and "zscore" not in name:
        return True
    return False


def _is_lookback(name: str, meta: Any) -> bool:
    if meta.feature_category == FeatureCategory.LOOKBACK:
        return True
    if _lookback_horizon(name) and name not in FEATURE_TO_CONTROLLER:
        if name.startswith("dgt_") or name in _SHARP_MOMENTUM:
            return False
        return True
    if name.startswith(("spot_body_pct_prev", "opt_body_pct_prev", "spot_range_pct_prev")):
        return True
    if name in (
        "atm_straddle_change_1m", "atm_straddle_change_5m",
        "atm_straddle_change_pct_1m", "atm_straddle_change_pct_5m",
        "atm_straddle_zscore_30m", "atm_straddle_zscore_change_5m",
        "atm_straddle_change_accel", "atm_straddle_slope_5m", "atm_straddle_slope_15m",
        "atm_straddle_pct_change_from_open",
        "chain_pcr_change_5m", "atm_pcr_change_5m",
    ):
        return True
    return False


def _lookback_warmup(name: str, meta: Any) -> tuple[str, str]:
    hz = _lookback_horizon(name)
    if hz:
        return "Calendar", hz
    eff = meta.effective_warmup_samples or meta.intrinsic_warmup_samples
    if eff:
        return "Sample", str(eff)
    if meta.intrinsic_warmup_sec:
        return "Calendar", f"{meta.intrinsic_warmup_sec}s"
    return "Calendar", "horizon"


def _classify_feature(name: str, group_id: str, meta: Any, judgments: list[str]) -> ClassificationRow:
    ctrl = FEATURE_TO_CONTROLLER.get(name)
    if ctrl:
        phase = CONTROLLER_PHASE.get(ctrl, "3")
        is_composite = ctrl in COMPOSITE_EMITTERS
        memory = "Controller-derived" if is_composite else "Controller"
        gap = _gap_policy_for(ctrl, memory)
        wtype, wval = CONTROLLER_WARMUP.get(ctrl, ("Sample", "0"))
        if ctrl in DERIVED_READERS:
            policy = "Derived"
        elif is_composite:
            policy = "ControllerDerived"
        else:
            policy = "NullUntilReady"
        depends = FEATURE_DEPENDS.get(name)
        if depends is None:
            if name.startswith("spot_ema") and name.endswith("_to_ltp_ratio"):
                depends = f"{ctrl} + LTP"
            elif name.startswith("ltp_to_spot_ema"):
                # Pipeline Owned packaging (Wave 5); kept for legacy name heuristics.
                depends = f"{ctrl} + spot_high + spot_low + LTP"
            elif ctrl.startswith("spot.hl."):
                # HL levels / channel width — no LTP required.
                depends = f"{ctrl} + spot_high + spot_low"
            else:
                depends = ctrl
        storage = _state_storage_for(ctrl, memory)
        sources = _source_controllers_label(ctrl)
        return ClassificationRow(
            feature=name, group_id=group_id, phase=phase,
            controller_owner=ctrl, future_owner="—",
            source_controllers=sources, memory_model=memory, gap_policy=gap,
            warmup_type=wtype, warmup_value=wval,
            emission_policy=policy, depends_on=depends,
            state_storage=storage,
        )

    if name in _SHARP_MOMENTUM:
        return ClassificationRow(
            name, group_id, "3", "—", "spot.momentum", "—",
            "Stateless", "None", "Immediate", "0",
            "AlwaysValid", "DayContext spot momentum snapshot",
            "DayContext snapshot",
        )

    if name in _TIME_FEATURES:
        return ClassificationRow(
            name, group_id, "n/a", "—", "—", "—",
            "Stateless", "None", "Immediate", "0",
            "AlwaysValid", "session clock + expiry calendar",
            "session_clock",
        )

    if name in _RAW_TICK or meta.feature_category == FeatureCategory.RAW:
        dep = "tick/row"
        if name == "spot":
            dep = "spot tick"
        elif name in ("delta", "gamma", "theta", "vega", "vanna", "volga", "charm", "speed"):
            dep = "greeks snapshot"
        return ClassificationRow(
            name, group_id, "n/a", "—", "—", "—",
            "Stateless", "None", "Immediate", "0",
            "Immediate", dep, "tick/row",
        )

    if _is_chain_snapshot(name, group_id):
        extra = " + OI" if "pcr" in name else (" + LTP" if "atm6" in name or name.startswith(("ce_", "pe_")) else "")
        return ClassificationRow(
            name, group_id, "n/a", "—", "ChainController", "—",
            "Stateless", "None", "Immediate", "0",
            "AlwaysValid", f"ChainMaps snapshot{extra}",
            "ChainMaps snapshot",
        )

    if _is_lookback(name, meta):
        hz = _lookback_horizon(name)
        scope = "token-scoped" if group_id not in ("chain", "atm_straddle") else "chain-scoped"
        if name.startswith(("chain_pcr", "atm_pcr", "atm_straddle")):
            scope = "chain-scoped"
        lb = f"prior rows (lookback {hz}, {scope}, gap-bounded)" if hz else "prior rows (lookback, gap-bounded)"
        wtype, wval = _lookback_warmup(name, meta)
        future = "ChainController" if scope == "chain-scoped" else "—"
        phase = "3" if future != "—" or name == "atm_straddle_zscore_30m" else "n/a"
        return ClassificationRow(
            name, group_id, phase, "—", future, "—",
            "Lookback", "StopAtGap", wtype, wval,
            "NullUntilReady", lb, "prior_rows_scan",
        )

    if name in ("ltp_to_dgt_reiv_ratio", "ltp_to_bs_reiv_ratio", "bs_reiv_to_ltp_ratio"):
        return ClassificationRow(
            name, group_id, "2", "—", "—", "—",
            "Stateless", "None", "Immediate", "0",
            "AlwaysValid", "bs_reiv_pred/dgt_reiv_pred + LTP",
            "tick/row",
        )

    if name == "iv_vs_atm":
        return ClassificationRow(
            name, group_id, "n/a", "—", "—", "—",
            "Stateless", "None", "Immediate", "0",
            "AlwaysValid", "IV + atm_straddle",
            "tick/row",
        )

    if meta.feature_category == FeatureCategory.ROLLING and name not in FEATURE_TO_CONTROLLER:
        j = f"{name}: policy ROLLING but no controller — pending phase assignment"
        judgments.append(j)
        return ClassificationRow(
            name, group_id, "3", "—", "TBD controller", "—",
            "Lookback", "StopAtGap", "Sample", str(meta.effective_warmup_samples or "?"),
            "NullUntilReady", "pending controller mapping",
            "prior_rows_scan", j,
        )

    wtype, wval = ("Immediate", "0")
    policy = "AlwaysValid"
    if meta.effective_warmup_samples or meta.intrinsic_warmup_sec:
        wtype, wval = _lookback_warmup(name, meta) if meta.intrinsic_warmup_sec else ("Sample", str(meta.effective_warmup_samples))
        policy = "NullUntilReady"

    deps = [d for d in meta.dependencies if not d.startswith("__roll.") and d not in ("timestamp", "token")]
    dep_str = " + ".join(deps) if deps else "point-in-time formula"

    return ClassificationRow(
        name, group_id, "n/a", "—", "—", "—",
        "Stateless", "None", wtype, wval,
        policy, dep_str, "tick/row",
    )


def _escape_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _render_table(rows: list[ClassificationRow]) -> str:
    header = (
        "| Feature | Phase | Controller owner | Future owner | Source controller(s) | "
        "Memory model | Gap policy | Warmup type | Warmup value | Emission policy | "
        "Depends on | State storage |"
    )
    sep = (
        "|---------|-------|------------------|--------------|----------------------|"
        "--------------|------------|-------------|--------------|-----------------|"
        "------------|---------------|"
    )
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| `{r.feature}` | {r.phase} | {r.controller_owner} | {r.future_owner} | "
            f"{_escape_cell(r.source_controllers)} | {r.memory_model} | {r.gap_policy} | "
            f"{r.warmup_type} | {r.warmup_value} | {r.emission_policy} | "
            f"{_escape_cell(r.depends_on)} | {_escape_cell(r.state_storage)} |"
        )
    return "\n".join(lines)


def generate_markdown() -> tuple[str, list[str], list[str]]:
    groups = _feature_groups()
    names = _all_features()
    if len(names) != 206:
        raise RuntimeError(f"Expected 206 features, got {len(names)}")

    reg = load_feature_policy_registry(feature_names=names)
    judgments: list[str] = []
    rows_by_group: dict[str, list[ClassificationRow]] = {}

    for name in names:
        gid = groups[name]
        meta = reg.get(name)
        if meta is None:
            judgments.append(f"{name}: missing policy metadata")
            meta = type("M", (), {
                "feature_category": FeatureCategory.DERIVED,
                "dependencies": (),
                "effective_warmup_samples": 0,
                "intrinsic_warmup_samples": 0,
                "intrinsic_warmup_sec": 0,
            })()
        row = _classify_feature(name, gid, meta, judgments)
        if row.judgment and row.judgment not in judgments:
            judgments.append(row.judgment)
        rows_by_group.setdefault(gid, []).append(row)

    counts = {"Controller": 0, "Controller-derived": 0, "Lookback": 0, "Stateless": 0}
    for name in names:
        gid = groups[name]
        meta = reg.get(name)
        if meta is None:
            meta = type("M", (), {"feature_category": FeatureCategory.DERIVED, "dependencies": ()})()
        mm = _classify_feature(name, gid, meta, []).memory_model
        counts[mm] = counts.get(mm, 0) + 1

    parts = [
        "# 240-Feature Classification Sheet",
        "",
        "```",
        generated_document_header(),
        "```",
        "",
        "Generated by `scripts/generate_feature_classification.py`. "
        "Bump `FEATURE_REGISTRY_VERSION` in `controller_registry.py` when the registry changes.",
        "",
        "Pre-implementation contract companion to [CONTROLLER_OWNERSHIP.md](CONTROLLER_OWNERSHIP.md) "
        "and [CONTROLLER_LIFECYCLE.md](CONTROLLER_LIFECYCLE.md). **Approved for sign-off**.",
        "",
        "## Governance (single source of truth)",
        "",
        "This sheet is the **authoritative registry** for controller ownership. Rules:",
        "",
        "1. Every feature appears **exactly once** (240 rows, no duplicates).",
        "2. Every stateful rolling feature (`Memory model` = `Controller`) has **exactly one** `Controller owner`.",
        "3. A feature may **read** from multiple controllers (`Source controller(s)`) but must **not own state** belonging to another controller.",
        "4. **Any new feature** must be added here (and `_REGISTRY_FEATURES`) **before** implementation.",
        "5. **Ownership changes** require regenerating this sheet **in the same commit** as the code change.",
        "",
        "**Generator also validates:** registry coverage, warmup consistency, build-phase order "
        "(sources must be same or earlier phase), dependency acyclicity. "
        "See [CONTROLLER_DEPENDENCY_GRAPH.md](CONTROLLER_DEPENDENCY_GRAPH.md).",
        "",
        "Lookback features are history-based (`Memory model` = `Lookback`); they are not controller-owned.",
        "",
        "## Column definitions",
        "",
        "| Column | Meaning |",
        "|--------|---------|",
        "| **Phase** | Controller migration phase (`1`/`2`/`3`/`n/a`) |",
        "| **Controller owner** | Rolling controller or composite emitter ID |",
        "| **Future owner** | Planned owner if not controller-owned today |",
        "| **Source controller(s)** | Controllers that must be updated before this feature can emit (build order) |",
        "| **Memory model** | `Controller` / `Controller-derived` / `Lookback` / `Stateless` |",
        "| **Gap policy** | `Reset` (token controller) / `Continue` (spot) / `StopAtGap` (lookback) / `None` |",
        "| **Warmup type** | `Sample` / `Calendar` / `Session` / `Immediate` |",
        "| **Warmup value** | Numeric samples or calendar horizon |",
        "| **Emission policy** | `NullUntilReady` / `ControllerDerived` / `Immediate` / `AlwaysValid` / `Derived` |",
        "| **Depends on** | Full explicit input list (controllers + LTP + spot + …) |",
        "| **State storage** | Where values live, or explicit `Reads: …` for derived emitters |",
        "",
        "## Lookback gap rule",
        "",
        "Lookback features use **Gap policy = StopAtGap**: scans stop at the most recent token gap "
        "(boundary enforced — not a controller reset).",
        "",
        "## Summary",
        "",
        f"- **Total features:** {len(names)}",
        f"- **Controller:** {counts.get('Controller', 0)}",
        f"- **Controller-derived:** {counts.get('Controller-derived', 0)}",
        f"- **Lookback:** {counts.get('Lookback', 0)}",
        f"- **Stateless:** {counts.get('Stateless', 0)}",
        "",
        "## Manual judgment calls",
        "",
    ]
    all_j = list(MANUAL_JUDGMENTS) + [j for j in judgments if j not in MANUAL_JUDGMENTS]
    for j in all_j:
        parts.append(f"- {j}")
    parts.append("")

    group_labels = {
        "price": "Price", "dgt_reiv": "DGT REIV", "ratio": "Ratio", "greeks": "Greeks",
        "iv": "IV", "iv_zscore": "IV Z-Score", "iv_ema_ratio": "IV EMA Ratio Features",
        "oi": "OI", "volume": "Volume",
        "momentum": "Momentum", "sharp_momentum": "Sharp Momentum", "spot_hl": "Spot HL",
        "time": "Time", "moneyness": "Moneyness", "ltp_to_spot": "LTP to Spot",
        "ltp_to_others": "LTP to Others", "spot_and_other_ratio": "Spot and Other Ratio",
        "atm_straddle": "ATM Straddle", "atm6_ltp": "ATM6 LTP", "chain": "Chain",
        "historical": "Historical", "advanced": "Advanced",
    }

    for gid in _REGISTRY_FEATURES:
        group_rows = rows_by_group.get(gid, [])
        if not group_rows:
            continue
        parts.extend([
            f"## {group_labels.get(gid, gid)} ({len(group_rows)})",
            "",
            _render_table(group_rows),
            "",
        ])

    return "\n".join(parts), names, judgments


def main() -> int:
    groups = _feature_groups()
    names = _all_features()
    reg = load_feature_policy_registry(feature_names=names)
    judgments: list[str] = []
    all_rows: list[ClassificationRow] = []
    for name in names:
        gid = groups[name]
        meta = reg.get(name)
        if meta is None:
            meta = type("M", (), {
                "feature_category": FeatureCategory.DERIVED,
                "dependencies": (),
                "effective_warmup_samples": 0,
                "intrinsic_warmup_samples": 0,
                "intrinsic_warmup_sec": 0,
            })()
        all_rows.append(_classify_feature(name, gid, meta, judgments))

    validate_governance(all_rows, names, FEATURE_TO_CONTROLLER)

    md, _, _ = generate_markdown()
    os.makedirs(_DOCS_CONTROLLERS, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(md)
    with open(DEP_GRAPH_PATH, "w", encoding="utf-8") as fh:
        fh.write(render_dependency_graph_markdown())
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {DEP_GRAPH_PATH}")
    print(f"Row count: {len(names)}")
    return 0 if len(names) == 206 else 1


if __name__ == "__main__":
    raise SystemExit(main())
