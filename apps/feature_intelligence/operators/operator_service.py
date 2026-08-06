"""Operator Registry service (Sprint 3)."""

from __future__ import annotations

import re
from pathlib import Path

from feature_intelligence.core.config import load_config
from feature_intelligence.operators.catalog import (
    OPERATOR_ID_PATTERN,
    compute_operator_catalog_hash,
)
from feature_intelligence.operators.operator_models import (
    CATEGORIES,
    COMPLEXITY_CLASSES,
    MISSING_DATA_POLICIES,
    WARMUP_POLICIES,
    OperatorRecord,
)
from feature_intelligence.operators.operator_store import OperatorStore
from feature_intelligence.registry.models import ValidationReport


class OperatorNotFoundError(KeyError):
    pass


class OperatorRegistryService:
    def __init__(self, db_path: Path | None = None) -> None:
        path = db_path or load_config().database.path
        self.db_path = Path(path)
        self._store = OperatorStore(self.db_path)

    def list_operators(self) -> list[OperatorRecord]:
        return self._store.list_all()

    def get_operator(self, operator_id: str) -> OperatorRecord:
        return self.get_by_id(operator_id)

    def get_by_id(self, operator_id: str) -> OperatorRecord:
        row = self._store.get_by_id(operator_id)
        if row is None:
            raise OperatorNotFoundError(operator_id)
        return row

    def get_by_name(self, canonical_name: str) -> OperatorRecord:
        row = self._store.get_by_name(canonical_name)
        if row is None:
            raise OperatorNotFoundError(canonical_name)
        return row

    def operator_exists(self, id_or_name: str) -> bool:
        if id_or_name.startswith("OP_"):
            return self._store.get_by_id(id_or_name) is not None
        return self._store.get_by_name(id_or_name) is not None

    def register_operator(self, record: OperatorRecord) -> OperatorRecord:
        """Internal / import only — not exposed on CLI."""
        if not re.fullmatch(OPERATOR_ID_PATTERN, record.operator_id):
            raise ValueError(f"Invalid operator_id: {record.operator_id}")
        if record.category not in CATEGORIES:
            raise ValueError(f"Invalid category: {record.category}")
        if record.warmup_policy not in WARMUP_POLICIES:
            raise ValueError(f"Invalid warmup_policy: {record.warmup_policy}")
        if record.missing_data_policy not in MISSING_DATA_POLICIES:
            raise ValueError(f"Invalid missing_data_policy: {record.missing_data_policy}")
        if record.complexity_class not in COMPLEXITY_CLASSES:
            raise ValueError(f"Invalid complexity_class: {record.complexity_class}")
        if self._store.get_by_id(record.operator_id) is not None:
            raise ValueError(f"Duplicate operator_id: {record.operator_id}")
        if self._store.get_by_name(record.canonical_name) is not None:
            raise ValueError(f"Duplicate canonical_name: {record.canonical_name}")
        return self._store.insert(record)

    def update_metadata(
        self,
        operator_id: str,
        *,
        display_name: str | None = None,
        description: str | None = None,
        extras_json: str | None = None,
    ) -> OperatorRecord:
        try:
            return self._store.update_metadata(
                operator_id,
                display_name=display_name,
                description=description,
                extras_json=extras_json,
            )
        except KeyError as exc:
            raise OperatorNotFoundError(operator_id) from exc

    def validate_registry(self) -> ValidationReport:
        from feature_intelligence.operators.operator_validation import validate_operators

        return validate_operators(self.db_path)

    def compute_operator_catalog_hash(self) -> str:
        return compute_operator_catalog_hash()
