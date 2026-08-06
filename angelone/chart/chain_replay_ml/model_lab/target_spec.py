"""Confidence TargetSpec registry — Market Outcomes + Replay-Based Outcomes.

Replay-Based targets are derived from continuous Confidence Label Builder
outcomes (net_pnl, return_pct, …). Adding a new binary threshold is a new
TargetSpec entry — not a new replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

Family = Literal["market", "replay_based"]
Generator = Literal["market_path", "strategy_replay"]

OutcomeRow = dict[str, Any]
RuleFn = Callable[[OutcomeRow], int | None]


@dataclass(frozen=True)
class TargetSpec:
    key: str
    label: str
    column: str
    family: Family
    generator: Generator
    """Human-readable rule for UI / docs."""
    rule: str = ""
    """Optional evaluator over a continuous replay-outcome row → 0|1|None."""
    derive: RuleFn | None = None

    @property
    def is_market(self) -> bool:
        return self.family == "market"

    @property
    def is_replay_based(self) -> bool:
        return self.family == "replay_based"


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _bin_gt(field: str, threshold: float) -> RuleFn:
    def _rule(row: OutcomeRow) -> int | None:
        v = _num(row.get(field))
        if v is None:
            return None
        return 1 if v > threshold else 0

    return _rule


def _bin_ge(field: str, threshold: float) -> RuleFn:
    def _rule(row: OutcomeRow) -> int | None:
        v = _num(row.get(field))
        if v is None:
            return None
        return 1 if v >= threshold else 0

    return _rule


def _bin_le(field: str, threshold: float) -> RuleFn:
    def _rule(row: OutcomeRow) -> int | None:
        v = _num(row.get(field))
        if v is None:
            return None
        return 1 if v <= threshold else 0

    return _rule


def _exit_reason_is(reason: str) -> RuleFn:
    want = str(reason).strip().lower()

    def _rule(row: OutcomeRow) -> int | None:
        raw = row.get("exit_reason")
        if raw is None:
            return None
        return 1 if str(raw).strip().lower() == want else 0

    return _rule


# ---------------------------------------------------------------------------
# Canonical specs
# ---------------------------------------------------------------------------

MARKET_TARGET_SPECS: tuple[TargetSpec, ...] = (
    TargetSpec(
        key="target_hit",
        label="Path Touch",
        column="target_reached",
        family="market",
        generator="market_path",
        rule="price path touches predicted LTP",
    ),
    TargetSpec(
        key="rr_1_1",
        label="RR 1:1",
        column="rr_1_1_hit",
        family="market",
        generator="market_path",
        rule="max profit / max drawdown ≥ 1.0 (path)",
    ),
    TargetSpec(
        key="rr_2_3",
        label="RR 2:3",
        column="rr_2_3_hit",
        family="market",
        generator="market_path",
        rule="max profit / max drawdown ≥ 0.666… (path)",
    ),
    TargetSpec(
        key="rr_1_2",
        label="RR 1:2",
        column="rr_1_2_hit",
        family="market",
        generator="market_path",
        rule="max profit / max drawdown ≥ 2.0 (path)",
    ),
    TargetSpec(
        key="rr_1_3",
        label="RR 1:3",
        column="rr_1_3_hit",
        family="market",
        generator="market_path",
        rule="max profit / max drawdown ≥ 3.0 (path)",
    ),
    TargetSpec(
        key="rr_1_4",
        label="RR 1:4",
        column="rr_1_4_hit",
        family="market",
        generator="market_path",
        rule="max profit / max drawdown ≥ 4.0 (path)",
    ),
)

# Replay-Based Outcomes — derived from Confidence Label Builder continuous fields.
REPLAY_TARGET_SPECS: tuple[TargetSpec, ...] = (
    TargetSpec(
        key="trade_winner",
        label="Trade Winner",
        column="trade_winner",
        family="replay_based",
        generator="strategy_replay",
        rule="net_pnl > 0",
        derive=_bin_gt("net_pnl", 0.0),
    ),
    TargetSpec(
        key="profit_100",
        label="Profit ≥ ₹100",
        column="profit_100",
        family="replay_based",
        generator="strategy_replay",
        rule="net_pnl >= 100",
        derive=_bin_ge("net_pnl", 100.0),
    ),
    TargetSpec(
        key="profit_250",
        label="Profit ≥ ₹250",
        column="profit_250",
        family="replay_based",
        generator="strategy_replay",
        rule="net_pnl >= 250",
        derive=_bin_ge("net_pnl", 250.0),
    ),
    TargetSpec(
        key="profit_500",
        label="Profit ≥ ₹500",
        column="profit_500",
        family="replay_based",
        generator="strategy_replay",
        rule="net_pnl >= 500",
        derive=_bin_ge("net_pnl", 500.0),
    ),
    TargetSpec(
        key="return_2",
        label="Return ≥ 2%",
        column="return_2",
        family="replay_based",
        generator="strategy_replay",
        rule="return_pct >= 2",
        derive=_bin_ge("return_pct", 2.0),
    ),
    TargetSpec(
        key="return_5",
        label="Return ≥ 5%",
        column="return_5",
        family="replay_based",
        generator="strategy_replay",
        rule="return_pct >= 5",
        derive=_bin_ge("return_pct", 5.0),
    ),
    TargetSpec(
        key="return_10",
        label="Return ≥ 10%",
        column="return_10",
        family="replay_based",
        generator="strategy_replay",
        rule="return_pct >= 10",
        derive=_bin_ge("return_pct", 10.0),
    ),
    TargetSpec(
        key="fast_60",
        label="Fast Winner ≤ 60s",
        column="fast_60",
        family="replay_based",
        generator="strategy_replay",
        rule="time_to_first_profit_sec <= 60",
        derive=_bin_le("time_to_first_profit_sec", 60.0),
    ),
    TargetSpec(
        key="fast_120",
        label="Fast Winner ≤ 120s",
        column="fast_120",
        family="replay_based",
        generator="strategy_replay",
        rule="time_to_first_profit_sec <= 120",
        derive=_bin_le("time_to_first_profit_sec", 120.0),
    ),
    TargetSpec(
        key="fast_300",
        label="Fast Winner ≤ 300s",
        column="fast_300",
        family="replay_based",
        generator="strategy_replay",
        rule="time_to_first_profit_sec <= 300",
        derive=_bin_le("time_to_first_profit_sec", 300.0),
    ),
    TargetSpec(
        key="target_exit",
        label="Target Exit",
        column="target_exit",
        family="replay_based",
        generator="strategy_replay",
        rule="exit_reason == target",
        derive=_exit_reason_is("target"),
    ),
)

ALL_TARGET_SPECS: tuple[TargetSpec, ...] = (*MARKET_TARGET_SPECS, *REPLAY_TARGET_SPECS)
TARGET_SPEC_BY_KEY: dict[str, TargetSpec] = {t.key: t for t in ALL_TARGET_SPECS}


def confidence_targets_for_manifest() -> tuple[dict[str, str], ...]:
    """Backward-compatible CONFIDENCE_TARGETS rows (key/column/label)."""
    out: list[dict[str, str]] = []
    for t in ALL_TARGET_SPECS:
        row: dict[str, str] = {"key": t.key, "column": t.column, "label": t.label}
        out.append(row)
    return tuple(out)


def market_label_columns() -> tuple[str, ...]:
    return tuple(t.column for t in MARKET_TARGET_SPECS)


def replay_label_columns() -> tuple[str, ...]:
    return tuple(t.column for t in REPLAY_TARGET_SPECS)


def all_label_columns() -> tuple[str, ...]:
    return tuple(t.column for t in ALL_TARGET_SPECS)


def inference_columns_for_key(model_key: str) -> dict[str, str]:
    """prediction_dataset column names for one confidence model key."""
    # Preserve legacy naming for market Path Touch / RR keys.
    legacy = {
        "target_hit": "target_hit",
        "rr_1_1": "rr_1_1",
        "rr_2_3": "rr_2_3",
        "rr_1_2": "rr_1_2",
        "rr_1_3": "rr_1_3",
        "rr_1_4": "rr_1_4",
    }
    slug = legacy.get(model_key, model_key)
    return {
        "pred": f"confidence_{slug}_pred",
        "model_id": f"confidence_{slug}_model_id",
        "threshold": f"confidence_{slug}_threshold",
        "created": f"confidence_{slug}_created",
    }


def build_inference_columns_map() -> dict[str, dict[str, str]]:
    return {t.key: inference_columns_for_key(t.key) for t in ALL_TARGET_SPECS}


def inference_core_columns() -> tuple[tuple[str, str], ...]:
    """(name, sql_type) pairs to ensure on prediction_dataset."""
    cols: list[tuple[str, str]] = []
    for t in ALL_TARGET_SPECS:
        ic = inference_columns_for_key(t.key)
        cols.append((ic["pred"], "INTEGER"))
        cols.append((ic["model_id"], "TEXT"))
        cols.append((ic["threshold"], "REAL"))
        cols.append((ic["created"], "TEXT"))
    return tuple(cols)


CONTINUOUS_OUTCOME_FIELDS: tuple[str, ...] = (
    "prediction_id",
    "net_pnl",
    "gross_pnl",
    "return_pct",
    "max_adverse_pct",
    "max_favorable_pct",
    "holding_seconds",
    "time_to_first_profit_sec",
    "exit_reason",
    "fees",
    "would_enter",
)


def derive_binary_labels(outcomes: list[OutcomeRow]) -> list[dict[str, Any]]:
    """Apply all Replay-Based TargetSpecs to continuous outcome rows."""
    specs = [t for t in REPLAY_TARGET_SPECS if t.derive is not None]
    out: list[dict[str, Any]] = []
    for row in outcomes:
        rec: dict[str, Any] = {"prediction_id": row.get("prediction_id")}
        for spec in specs:
            assert spec.derive is not None
            rec[spec.column] = spec.derive(row)
        out.append(rec)
    return out
