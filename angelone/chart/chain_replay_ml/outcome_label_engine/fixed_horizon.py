"""Fixed Horizon labeling strategy — migrated Stage 5 / backfill math (zero behavior change)."""

from __future__ import annotations

from typing import Any

from chain_replay_ml.dataset_builder.feature_plugins import horizon_column_name

from .registry import register_strategy
from .types import (
    SOURCE_MASTER,
    LabelBatchResult,
    LabelSourceContext,
    LabelStrategyConfig,
    StrategyCapabilities,
    StrategyMetadata,
    TargetDefinitions,
    validate_config_against_schema,
)

STRATEGY_ID = "fixed_horizon"
STRATEGY_VERSION = "1.0"

_METADATA = StrategyMetadata(
    strategy_id=STRATEGY_ID,
    version=STRATEGY_VERSION,
    display_name="Fixed Horizon",
    description="Future premium at fixed horizon",
    category="Regression",
)

_CAPABILITIES = StrategyCapabilities(
    strategy_id=STRATEGY_ID,
    supported_sources=frozenset({SOURCE_MASTER}),
    supported_problem_types=frozenset({"regression", "binary_classification"}),
)


def compute_fixed_horizon_targets(
    *,
    ts: float,
    opt_tl: Any,
    horizons_sec: list[int],
    max_stale_sec: float,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Exact Stage 5 / former ``compute_targets_for_row`` semantics.

    For each horizon: ``future_ts = ts + h``; if ``opt_tl.is_fresh_at(future_ts, max_stale_sec)``
    write ``opt_tl.ltp_rupees_at(future_ts)``, else ``None``.
    """
    col_to_horizon = {horizon_column_name(h): h for h in horizons_sec}
    target_cols = list(columns) if columns is not None else list(col_to_horizon.keys())
    out: dict[str, Any] = {}
    for col in target_cols:
        h = col_to_horizon.get(col)
        if h is None:
            continue
        future_ts = ts + float(h)
        if opt_tl and opt_tl.is_fresh_at(future_ts, max_stale_sec):
            out[col] = opt_tl.ltp_rupees_at(future_ts)
        else:
            out[col] = None
    return out


def _default_horizons_sec() -> list[int]:
    return [3, 5, 10, 30, 60, 180, 300]


class FixedHorizonStrategy:
    """Master / timeline–backed Fixed Horizon labels (``future_ltp_*``)."""

    @property
    def metadata(self) -> StrategyMetadata:
        return _METADATA

    @property
    def capabilities(self) -> StrategyCapabilities:
        return _CAPABILITIES

    def get_config_schema(self) -> dict[str, Any]:
        return {
            "horizons_sec": {
                "type": "int_list",
                "default": _default_horizons_sec(),
            },
            "max_stale_sec": {
                "type": "float",
                "default": 10.0,
            },
        }

    def get_target_definitions(self) -> TargetDefinitions:
        # Primary regression target remains the familiar 5m column; binary label_up_*
        # stay derived downstream of future_ltp_5m (export path unchanged).
        return TargetDefinitions(
            primary_target="future_ltp_5m",
            display_target=None,
            label_encoding=None,
        )

    def build_labels(
        self,
        source: LabelSourceContext,
        samples: Any,
        config: LabelStrategyConfig,
    ) -> LabelBatchResult:
        params = validate_config_against_schema(
            dict(config.params),
            self.get_config_schema(),
        )
        horizons_sec = [int(h) for h in (params.get("horizons_sec") or [])]
        max_stale_sec = float(params.get("max_stale_sec", 10.0))
        columns = params.get("columns")
        if columns is not None:
            columns = [str(c) for c in columns]

        target_columns = (
            list(columns)
            if columns is not None
            else [horizon_column_name(h) for h in horizons_sec]
        )
        rows_out: list[dict[str, Any]] = []
        for sample in samples or []:
            ts = float(sample["timestamp"])
            opt_tl = sample.get("_opt_tl")
            if opt_tl is None:
                opt_tl = sample.get("opt_tl")
            targets = compute_fixed_horizon_targets(
                ts=ts,
                opt_tl=opt_tl,
                horizons_sec=horizons_sec,
                max_stale_sec=max_stale_sec,
                columns=columns,
            )
            row = dict(sample)
            # Do not persist opaque timeline handles in labeled output.
            row.pop("_opt_tl", None)
            row.pop("opt_tl", None)
            row.update(targets)
            row.setdefault("is_valid", True)
            row.setdefault("invalid_reason", None)
            rows_out.append(row)

        return LabelBatchResult(
            rows=rows_out,
            target_columns=target_columns,
            target_definitions=self.get_target_definitions(),
            metadata={
                "strategy": STRATEGY_ID,
                "day": source.day,
                "source_kind": source.source_kind,
                "horizons_sec": horizons_sec,
                "max_stale_sec": max_stale_sec,
            },
        )


_FIXED_HORIZON = FixedHorizonStrategy()


def get_fixed_horizon_strategy() -> FixedHorizonStrategy:
    return _FIXED_HORIZON


def register_fixed_horizon_strategy(*, replace: bool = True) -> FixedHorizonStrategy:
    """Ensure Fixed Horizon is in the OLE registry."""
    register_strategy(_FIXED_HORIZON, replace=replace)
    return _FIXED_HORIZON
