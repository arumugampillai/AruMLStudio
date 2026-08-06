"""Feature transformation framework.

Lag, Difference, Difference Clip, Return, Anchor Return, Rolling Statistics,
Exponential Rolling, OHLC Aggregation, Interaction, Math (Unary),
Normalization, Regime / Bucket, and Derived are config-driven (disabled by default).

Time semantics: row-based only. Horizons must be exact multiples of the
dataset sample interval. Calendar-time lookups are not supported.
"""
from __future__ import annotations

from .anchor_return import anchor_return_column_name
from .base import FeatureTransformation, TransformationResult, TransformContext
from .config import (
    TRANSFORMATION_PIPELINE_VERSION,
    config_for_metadata,
    default_transformation_config,
    normalize_transformation_config,
)
from .describe import (
    MASTER_STAGE_ID,
    OutputDescriptor,
    PipelineDescription,
    StageDescriptor,
    describe_pipeline_stages,
)
from .difference import difference_column_name
from .difference_clip import difference_clip_column_name
from .exponential_rolling import exponential_rolling_column_name
from .interaction import interaction_column_name, normalize_interaction_op
from .math_transform import math_column_name
from .normalization import normalization_column_name
from .ohlc_aggregation import ohlc_aggregation_column_name
from .pipeline import (
    PipelineResult,
    describe_pipeline,
    format_pipeline_log_lines,
    run_transformation_pipeline,
    run_transformation_pipeline_on_parquet,
)
from .regime import regime_column_name
from .registry import (
    ensure_builtin_transformations,
    get_transformation,
    list_registered_transformations,
    registered_transformation_count,
    register_transformation,
)
from .return_transform import return_column_name
from .rolling import rolling_column_name
from .rolling_ohlc import rolling_ohlc_column_name
from .rolling_statistics import rolling_stat_column_name

__all__ = [
    "FeatureTransformation",
    "TransformationResult",
    "TransformContext",
    "PipelineResult",
    "PipelineDescription",
    "StageDescriptor",
    "OutputDescriptor",
    "MASTER_STAGE_ID",
    "TRANSFORMATION_PIPELINE_VERSION",
    "config_for_metadata",
    "default_transformation_config",
    "normalize_transformation_config",
    "describe_pipeline",
    "describe_pipeline_stages",
    "format_pipeline_log_lines",
    "run_transformation_pipeline",
    "run_transformation_pipeline_on_parquet",
    "ensure_builtin_transformations",
    "get_transformation",
    "list_registered_transformations",
    "registered_transformation_count",
    "register_transformation",
    "anchor_return_column_name",
    "difference_column_name",
    "difference_clip_column_name",
    "exponential_rolling_column_name",
    "interaction_column_name",
    "normalize_interaction_op",
    "math_column_name",
    "normalization_column_name",
    "ohlc_aggregation_column_name",
    "regime_column_name",
    "return_column_name",
    "rolling_column_name",
    "rolling_ohlc_column_name",
    "rolling_stat_column_name",
]
