"""Structured progress events for /ws/ml-dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


STAGE_NAMES = {
    1: "Load Database",
    2: "Validate Sources",
    3: "Sampling Grid",
    4: "Strike Selection",
    5: "Prediction Targets",
    6: "Feature Generation",
    7: "Dataset Validation",
    8: "Write Parquet",
}

TOTAL_STAGES = 8


@dataclass
class BuildProgress:
    job_id: str
    stage: int = 1
    stage_name: str = "Load Database"
    current: int = 0
    total: int = 0
    percent: float = 0.0
    message: str = ""
    rows: int = 0
    source_ticks: int = 0
    sample_points: int = 0
    source_day_index: int | None = None
    source_day_total: int | None = None
    substage: str | None = None
    sub_current: int | None = None
    sub_total: int | None = None
    substage_percent: float | None = None
    warnings: list[str] = field(default_factory=list)
    status: str = "running"  # running | completed | failed | cancelled
    output_parquet: str | None = None
    output_json: str | None = None
    output_parquet_bytes: int | None = None
    output_json_bytes: int | None = None
    output_expected_json: str | None = None
    error: str | None = None
    validation_checks: list[dict[str, Any]] = field(default_factory=list)
    current_check: str | None = None
    pipeline: dict[str, Any] | None = None
    dataset_stats: dict[str, Any] | None = None
    feature_groups_done: int | None = None
    feature_groups_total: int | None = None
    feature_groups_remaining: int | None = None
    feature_group_id: str | None = None
    feature_group_current: str | None = None
    # Target-horizon backfill runs *before* the numbered stage pipeline (it
    # reprocesses existing master days) but is tracked under stage 1 so its
    # elapsed time is visible. These fields let the UI distinguish "backfill
    # of existing data" from "loading a newly selected day" without either
    # of them stealing the other's stage name / progress counters.
    backfill_active: bool = False
    backfill_days_current: int | None = None
    backfill_days_total: int | None = None
    backfill_rows_current: int | None = None
    backfill_rows_total: int | None = None
    backfill_percent: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_stages"] = TOTAL_STAGES
        return d

    def emit(
        self,
        *,
        stage: int | None = None,
        stage_name: str | None = None,
        current: int | None = None,
        total: int | None = None,
        message: str | None = None,
        rows: int | None = None,
        source_ticks: int | None = None,
        sample_points: int | None = None,
        source_day_index: int | None = None,
        source_day_total: int | None = None,
        substage: str | None = None,
        sub_current: int | None = None,
        sub_total: int | None = None,
        clear_substage: bool = False,
        validation_checks: list[dict[str, Any]] | None = None,
        current_check: str | None = None,
        pipeline: dict[str, Any] | None = None,
        feature_groups_done: int | None = None,
        feature_groups_total: int | None = None,
        feature_groups_remaining: int | None = None,
        feature_group_id: str | None = None,
        feature_group_current: str | None = None,
        backfill_active: bool | None = None,
        backfill_days_current: int | None = None,
        backfill_days_total: int | None = None,
        backfill_rows_current: int | None = None,
        backfill_rows_total: int | None = None,
        backfill_percent: float | None = None,
    ) -> dict[str, Any]:
        if stage is not None:
            self.stage = stage
            self.stage_name = stage_name if stage_name is not None else STAGE_NAMES.get(stage, f"Stage {stage}")
            if stage != 6:
                self.feature_groups_done = None
                self.feature_groups_total = None
                self.feature_groups_remaining = None
                self.feature_group_id = None
                self.feature_group_current = None
        elif stage_name is not None:
            self.stage_name = stage_name
        if backfill_active is not None:
            self.backfill_active = backfill_active
            if not backfill_active:
                self.backfill_days_current = None
                self.backfill_days_total = None
                self.backfill_rows_current = None
                self.backfill_rows_total = None
                self.backfill_percent = None
        if backfill_days_current is not None:
            self.backfill_days_current = backfill_days_current
        if backfill_days_total is not None:
            self.backfill_days_total = backfill_days_total
        if backfill_rows_current is not None:
            self.backfill_rows_current = backfill_rows_current
        if backfill_rows_total is not None:
            self.backfill_rows_total = backfill_rows_total
        if backfill_percent is not None:
            self.backfill_percent = backfill_percent
        if current is not None:
            self.current = current
        if total is not None:
            self.total = total
        if message is not None:
            self.message = message
        if rows is not None:
            self.rows = rows
        if source_ticks is not None:
            self.source_ticks = source_ticks
        if sample_points is not None:
            self.sample_points = sample_points
        if source_day_index is not None:
            self.source_day_index = source_day_index
        if source_day_total is not None:
            self.source_day_total = source_day_total
        if validation_checks is not None:
            self.validation_checks = validation_checks
        if current_check is not None:
            self.current_check = current_check
        if pipeline is not None:
            self.pipeline = pipeline
        if feature_groups_done is not None:
            self.feature_groups_done = feature_groups_done
        if feature_groups_total is not None:
            self.feature_groups_total = feature_groups_total
        if feature_groups_remaining is not None:
            self.feature_groups_remaining = feature_groups_remaining
        if feature_group_id is not None:
            self.feature_group_id = feature_group_id
        if feature_group_current is not None:
            self.feature_group_current = feature_group_current
        if clear_substage:
            self.substage = None
            self.sub_current = None
            self.sub_total = None
            self.substage_percent = None
            self.feature_groups_done = None
            self.feature_groups_total = None
            self.feature_groups_remaining = None
            self.feature_group_id = None
            self.feature_group_current = None
        else:
            if substage is not None:
                self.substage = substage
            if sub_current is not None:
                self.sub_current = sub_current
            if sub_total is not None:
                self.sub_total = sub_total
        if self.total > 0:
            intra = 0.0
            if self.sub_total and self.sub_current is not None and self.sub_total > 0:
                intra = min(1.0, self.sub_current / self.sub_total)
            self.percent = round(100.0 * (self.current + intra) / self.total, 1)
        else:
            self.percent = 0.0
        if self.sub_total and self.sub_current is not None and self.sub_total > 0 and self.stage != 7:
            self.substage_percent = round(100.0 * self.sub_current / self.sub_total, 1)
        else:
            self.substage_percent = None
        return self.to_dict()


def feature_group_progress_fields(
    *,
    groups_done: int,
    groups_total: int,
    group_id: str | None = None,
    group_label: str | None = None,
    row_current: int | None = None,
    row_total: int | None = None,
) -> dict[str, Any]:
    """Standard progress fields for feature-group build updates."""
    remaining = max(0, int(groups_total) - int(groups_done))
    out: dict[str, Any] = {
        "feature_groups_done": int(groups_done),
        "feature_groups_total": int(groups_total),
        "feature_groups_remaining": remaining,
    }
    if group_id:
        out["feature_group_id"] = str(group_id)
    if group_label:
        out["feature_group_current"] = str(group_label)
    if row_current is not None and row_total is not None:
        out["sub_current"] = int(row_current)
        out["sub_total"] = int(row_total)
    return out
