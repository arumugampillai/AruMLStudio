"""Chain-context enrichment for current-to-ATM6 flow ratio (master row-build parity)."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from path_config import CHART_DATA_ROOT as _CHART_DIR

from ..current_to_atm6_flow import (
    CURRENT_TO_ATM6_FLOW_FEATURE,
    enrich_current_to_atm6_flow_dataframe,
)
from .base import FeatureTransformation, TransformationResult, TransformContext
from .registry import register_transformation
from .time_shift import LagConfigError, resolve_sample_interval, resolve_transform_params


@register_transformation
class CurrentToAtm6FlowTransformation(FeatureTransformation):
    """Replay chain maps to materialize ``current_to_atm6_flow_delta_ltp_to_spot_ratio``."""

    id = "current_to_atm6_flow"
    name = "Current ATM6 Flow"
    order = 41
    enabled = False
    depends_on: list[str] = []
    params: dict[str, Any] = {}

    def describe(
        self,
        params: dict[str, Any] | None = None,
        *,
        upstream=None,
        master_features: list[str] | None = None,
        sample_interval_sec: float | int | None = None,
        enabled: bool | None = None,
    ):
        from .describe import MASTER_STAGE_ID, OutputDescriptor, make_stage_descriptor

        del upstream, master_features
        is_enabled = bool(self.enabled if enabled is None else enabled)
        params = dict(params or {})
        column = str(params.get("column") or CURRENT_TO_ATM6_FLOW_FEATURE).strip()
        outputs = [
            OutputDescriptor(
                name=column,
                kind="chain_enrich",
                source_feature="chain_maps",
                op="current_to_atm6_flow",
            )
        ]
        return make_stage_descriptor(
            self,
            enabled=is_enabled,
            outputs=outputs,
            input_sources=[MASTER_STAGE_ID],
            notes="Chain replay enrichment for ATM6 wing flow ratio.",
        )

    def transform(self, df: pd.DataFrame, context: TransformContext) -> TransformationResult:
        t0 = time.perf_counter()
        params = resolve_transform_params(self.id, getattr(self, "params", None), context.config)
        interval = resolve_sample_interval(context, params)
        column = str(params.get("column") or CURRENT_TO_ATM6_FLOW_FEATURE).strip()
        if not column:
            raise LagConfigError(
                "Current ATM6 Flow Transformation\n"
                "params.column is required."
            )
        if column in df.columns:
            elapsed = round(max(time.perf_counter() - t0, 0.0), 4)
            return TransformationResult(
                frame=df,
                created_columns=[],
                elapsed_sec=elapsed,
                rows_processed=int(len(df)),
                transformation_id=self.id,
                transformation_name=self.name,
                messages=[f"Column already present: {column}"],
            )
        if context.cancelled():
            return TransformationResult.passthrough(
                df,
                transformation_id=self.id,
                transformation_name=self.name,
            )

        market = str((context.dataset_info or {}).get("market") or "NIFTY").upper()
        chart_dir = str(_CHART_DIR)
        try:
            out = enrich_current_to_atm6_flow_dataframe(
                df,
                chart_dir=chart_dir,
                market=market,
                feature_grid_step_sec=int(interval) if float(interval).is_integer() else int(round(interval)),
                column=column,
            )
        except ValueError as exc:
            raise LagConfigError(
                "Current ATM6 Flow Transformation\n"
                + str(exc)
            ) from exc

        elapsed = round(max(time.perf_counter() - t0, 0.0), 4)
        context.log("Current ATM6 Flow Transformation")
        context.log(f"Columns Created : 1")
        context.log(f"Elapsed : {elapsed:.2f} s")
        return TransformationResult(
            frame=out,
            created_columns=[column],
            elapsed_sec=elapsed,
            rows_processed=int(len(out)),
            transformation_id=self.id,
            transformation_name=self.name,
            messages=[
                f"column={column}",
                f"sample_interval_sec={interval}",
                "frame_backend=chain_replay",
            ],
        )
