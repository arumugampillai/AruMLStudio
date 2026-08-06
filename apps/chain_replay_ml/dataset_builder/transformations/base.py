"""Transformation interface, context, and per-transform result."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .describe import PipelineDescription, StageDescriptor


@dataclass
class TransformContext:
    """Shared runtime context for every transformation.

    Grow this object instead of changing transform signatures when Lag / Rolling /
    EMA need sample interval, warmup, logging, etc.
    """

    config: dict[str, Any] = field(default_factory=dict)
    # Dataset / build identity
    data_dir: str | None = None
    dataset_name: str | None = None
    dataset_info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Timing / policy (seconds unless noted)
    sample_interval_sec: float | None = None
    warmup_seconds: float | None = None
    prediction_minutes: float | None = None
    # Observability / control
    logger: Callable[[str], None] | None = None
    progress_callback: Callable[[str, int, int], None] | None = None
    cancel_token: Callable[[], bool] | None = None
    # Escape hatch for future fields without signature churn
    extras: dict[str, Any] = field(default_factory=dict)

    def log(self, message: str) -> None:
        if self.logger is None:
            return
        try:
            self.logger(str(message))
        except Exception:
            pass

    def cancelled(self) -> bool:
        if self.cancel_token is None:
            return False
        try:
            return bool(self.cancel_token())
        except Exception:
            return False

    def report_progress(self, message: str, current: int = 0, total: int = 0) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(str(message), int(current or 0), int(total or 0))
        except Exception:
            pass


@dataclass
class TransformationResult:
    """Outcome of a single transformation step."""

    frame: pd.DataFrame
    created_columns: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0
    rows_processed: int = 0
    transformation_id: str = ""
    transformation_name: str = ""
    messages: list[str] = field(default_factory=list)

    @classmethod
    def passthrough(
        cls,
        df: pd.DataFrame,
        *,
        transformation_id: str = "",
        transformation_name: str = "",
        elapsed_sec: float = 0.0,
    ) -> TransformationResult:
        return cls(
            frame=df,
            created_columns=[],
            elapsed_sec=float(elapsed_sec),
            rows_processed=int(len(df)),
            transformation_id=transformation_id,
            transformation_name=transformation_name,
        )


class FeatureTransformation(ABC):
    """Base class for all feature transformations.

    Dataset Builder executes registered instances without knowing whether a
    transform is Lag, Difference, EMA, etc.
    """

    id: str = ""
    name: str = ""
    order: int = 100
    # Class-level default; runtime enablement comes from configuration.
    enabled: bool = False
    # Declared dependencies (by transform id). Pipeline may enforce / order by these.
    depends_on: list[str] = []

    @abstractmethod
    def transform(self, df: pd.DataFrame, context: TransformContext) -> TransformationResult:
        """Apply this transformation and report created columns / timing."""

    def describe(
        self,
        params: dict[str, Any] | None = None,
        *,
        upstream: "PipelineDescription | None" = None,
        master_features: list[str] | None = None,
        sample_interval_sec: float | int | None = None,
        enabled: bool | None = None,
    ) -> "StageDescriptor":
        """Plan-time self-description for this stage.

        Default: identity + empty outputs. Concrete transforms override to list
        planned ``OutputDescriptor``s from ``params``.
        """
        from .describe import MASTER_STAGE_ID, make_stage_descriptor

        del upstream, master_features, sample_interval_sec, params
        return make_stage_descriptor(
            self,
            enabled=bool(self.enabled if enabled is None else enabled),
            outputs=[],
            input_sources=[MASTER_STAGE_ID],
        )

    def planned_outputs(
        self,
        params: dict[str, Any] | None = None,
        *,
        upstream: "PipelineDescription | None" = None,
        master_features: list[str] | None = None,
        sample_interval_sec: float | int | None = None,
        enabled: bool | None = None,
    ) -> list[str]:
        """Column names this stage would create (derived from ``describe()``)."""
        return self.describe(
            params,
            upstream=upstream,
            master_features=master_features,
            sample_interval_sec=sample_interval_sec,
            enabled=enabled,
        ).output_names

    def __repr__(self) -> str:
        deps = ",".join(self.depends_on) if self.depends_on else ""
        return (
            f"{self.__class__.__name__}(id={self.id!r}, order={self.order}, "
            f"enabled={self.enabled}, depends_on=[{deps}])"
        )
