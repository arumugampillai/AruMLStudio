"""Primitive Catalog service APIs (Sprint 1)."""

from __future__ import annotations

from pathlib import Path

from feature_intelligence.core.config import load_config
from feature_intelligence.registry.catalog import (
    DATA_SOURCES,
    UNITS,
    compute_seed_catalog_hash,
)
from feature_intelligence.registry.models import PrimitiveRecord, ValidationReport
from feature_intelligence.registry.store import PrimitiveStore
from feature_intelligence.registry.validation import validate_primitives


class PrimitiveNotFoundError(KeyError):
    """Raised when a primitive_id / name is not in the catalog."""


class PrimitiveCatalogService:
    """In-process Primitive Catalog API (curated seed; no runtime register)."""

    def __init__(self, db_path: Path | None = None) -> None:
        path = db_path or load_config().database.path
        self.db_path = Path(path)
        self._store = PrimitiveStore(self.db_path)

    def list_primitives(self) -> list[PrimitiveRecord]:
        return self._store.list_all()

    def get_primitive(self, primitive_id: str) -> PrimitiveRecord:
        row = self._store.get_by_id(primitive_id)
        if row is None:
            raise PrimitiveNotFoundError(primitive_id)
        return row

    def get_primitive_by_name(self, name: str) -> PrimitiveRecord:
        row = self._store.get_by_name(name)
        if row is None:
            raise PrimitiveNotFoundError(name)
        return row

    def primitive_exists(self, primitive_id: str) -> bool:
        return self._store.get_by_id(primitive_id) is not None

    def update_primitive_metadata(
        self,
        primitive_id: str,
        *,
        description: str | None = None,
        data_source: str | None = None,
        units: str | None = None,
    ) -> PrimitiveRecord:
        if data_source is not None and data_source not in DATA_SOURCES:
            raise ValueError(f"Invalid data_source: {data_source}")
        if units is not None and units not in UNITS:
            raise ValueError(f"Invalid units: {units}")
        if description is None and data_source is None and units is None:
            raise ValueError("No metadata fields provided to update")
        return self._store.update_metadata(
            primitive_id,
            description=description,
            data_source=data_source,
            units=units,
        )

    def validate_primitives(self) -> ValidationReport:
        return validate_primitives(self.db_path)

    def compute_seed_catalog_hash(self) -> str:
        return compute_seed_catalog_hash()


def get_default_service(db_path: Path | None = None) -> PrimitiveCatalogService:
    return PrimitiveCatalogService(db_path=db_path)
