"""Metadata-driven dataset preview — delegates to DatasetSelectionEngine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dataset_selection_engine import (
    DatasetSelectionEngine,
    DatasetSelectionSpec,
    SelectionPreviewResult,
)


@dataclass
class PreviewFilters:
    """Backward-compatible preview filter bag — prefer DatasetSelectionSpec."""

    market: str = "NIFTY"
    master_db_path: str | None = None
    interval_sec: int | None = None
    selected_days: list[str] = field(default_factory=list)
    premium_range: dict[str, float] | None = None
    atm_band: int | None = None
    delta_range: dict[str, float] | None = None

    def to_spec(self) -> DatasetSelectionSpec:
        body: dict[str, Any] = {
            "market": self.market,
            "interval_sec": self.interval_sec,
            "master_dataset": self.master_db_path,
            "selected_days": self.selected_days,
            "atm_band": self.atm_band,
        }
        if self.premium_range:
            body["premium_range"] = self.premium_range
        if self.delta_range:
            body["delta_range"] = self.delta_range
        return DatasetSelectionSpec.from_api_body(body)


class MasterDatasetPreviewService:
    """Estimate filtered dataset size from metadata only (no samples scan)."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    @classmethod
    def resolve_db_path(cls, data_dir: str, **kwargs: Any) -> str:
        spec = DatasetSelectionSpec.from_api_body({
            "market": kwargs.get("market", "NIFTY"),
            "interval_sec": kwargs.get("interval_sec"),
            "master_dataset": kwargs.get("master_dataset"),
        })
        return DatasetSelectionEngine.resolve_master_db_path(data_dir, spec)

    def preview(self, filters: PreviewFilters) -> dict[str, Any]:
        engine = DatasetSelectionEngine(filters.to_spec(), self.db_path)
        return engine.preview().to_api_dict()

    @classmethod
    def preview_from_body(cls, data_dir: str, body: dict[str, Any]) -> dict[str, Any]:
        return DatasetSelectionEngine.preview_from_body(data_dir, body)


__all__ = [
    "MasterDatasetPreviewService",
    "PreviewFilters",
    "DatasetSelectionEngine",
    "DatasetSelectionSpec",
    "SelectionPreviewResult",
]
