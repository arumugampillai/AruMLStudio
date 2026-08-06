"""Feature Registry service APIs (Sprint 2)."""

from __future__ import annotations

import re
from pathlib import Path

from feature_intelligence.core.config import load_config
from feature_intelligence.registry.feature_definition_hash import compute_definition_hash
from feature_intelligence.registry.feature_ids import (
    generate_feature_uuid,
    normalize_feature_uuid,
    normalize_transformation_uuid,
)
from feature_intelligence.registry.feature_models import (
    CANONICAL_NAME_PATTERN,
    GAP_POLICIES,
    MEMORY_MODELS,
    RESEARCH_STATES,
    FeatureRecord,
    SyncReport,
)
from feature_intelligence.registry.feature_store import FeatureStore
from feature_intelligence.registry.models import ValidationReport
from feature_intelligence.registry.store import PrimitiveStore


class FeatureNotFoundError(KeyError):
    pass


class FeatureRegistryService:
    def __init__(self, db_path: Path | None = None) -> None:
        path = db_path or load_config().database.path
        self.db_path = Path(path)
        self._store = FeatureStore(self.db_path)
        self._primitives = PrimitiveStore(self.db_path)

    def register_feature(
        self,
        *,
        canonical_name: str,
        display_name: str,
        primitive_ids: list[str],
        created_by: str,
        controller_owner: str,
        warmup_periods: int,
        gap_policy: str,
        memory_model: str,
        definition_version: str = "1.0",
        implementation_version: str = "1",
        research_state: str = "EXPERIMENTAL",
        description: str | None = None,
        feature_version: str | None = None,
        transformation_uuid: str | None = None,
        legacy_feature_id: str | None = None,
        feature_uuid: str | None = None,
    ) -> FeatureRecord:
        if not re.fullmatch(CANONICAL_NAME_PATTERN, canonical_name):
            raise ValueError(f"Invalid canonical_name: {canonical_name!r}")
        if not primitive_ids:
            raise ValueError("At least one primitive_id is required")
        if gap_policy not in GAP_POLICIES:
            raise ValueError(f"Invalid gap_policy: {gap_policy}")
        if memory_model not in MEMORY_MODELS:
            raise ValueError(f"Invalid memory_model: {memory_model}")
        if research_state not in RESEARCH_STATES:
            raise ValueError(f"Invalid research_state: {research_state}")
        if not definition_version or not implementation_version:
            raise ValueError("definition_version and implementation_version required")

        for pid in primitive_ids:
            if self._primitives.get_by_id(pid) is None:
                raise ValueError(f"Unknown primitive_id: {pid}")

        if self._store.get_by_name(canonical_name) is not None:
            raise ValueError(f"Duplicate canonical_name: {canonical_name}")

        fu = (
            normalize_feature_uuid(feature_uuid)
            if feature_uuid
            else generate_feature_uuid()
        )
        if self._store.get_by_uuid(fu) is not None:
            raise ValueError(f"Duplicate feature_uuid: {fu}")

        tr = normalize_transformation_uuid(transformation_uuid)
        dhash = compute_definition_hash(
            canonical_name=canonical_name,
            warmup_periods=warmup_periods,
            gap_policy=gap_policy,
            memory_model=memory_model,
            primitive_ids=primitive_ids,
        )
        record = FeatureRecord(
            feature_uuid=fu,
            canonical_name=canonical_name,
            display_name=display_name,
            definition_version=definition_version,
            implementation_version=implementation_version,
            feature_version=feature_version,
            definition_hash=dhash,
            transformation_uuid=tr,
            legacy_feature_id=legacy_feature_id,
            description=description,
            created_by=created_by,
            controller_owner=controller_owner,
            warmup_periods=int(warmup_periods),
            gap_policy=gap_policy,
            memory_model=memory_model,
            research_state=research_state,
            primitive_ids=tuple(primitive_ids),
        )
        return self._store.insert(record, primitive_ids=list(primitive_ids))

    def get_feature(self, feature_uuid: str) -> FeatureRecord:
        return self.get_by_uuid(feature_uuid)

    def get_by_uuid(self, feature_uuid: str) -> FeatureRecord:
        row = self._store.get_by_uuid(normalize_feature_uuid(feature_uuid))
        if row is None:
            raise FeatureNotFoundError(feature_uuid)
        return row

    def get_by_name(self, canonical_name: str) -> FeatureRecord:
        row = self._store.get_by_name(canonical_name)
        if row is None:
            raise FeatureNotFoundError(canonical_name)
        return row

    def feature_exists(self, uuid_or_name: str) -> bool:
        if uuid_or_name.startswith("FEAT_"):
            try:
                return self._store.get_by_uuid(normalize_feature_uuid(uuid_or_name)) is not None
            except ValueError:
                return False
        return self._store.get_by_name(uuid_or_name) is not None

    def update_metadata(
        self,
        feature_uuid: str,
        *,
        display_name: str | None = None,
        description: str | None = None,
        controller_owner: str | None = None,
        research_state: str | None = None,
        implementation_version: str | None = None,
    ) -> FeatureRecord:
        fu = normalize_feature_uuid(feature_uuid)
        if research_state is not None and research_state not in RESEARCH_STATES:
            raise ValueError(f"Invalid research_state: {research_state}")
        try:
            return self._store.update_metadata(
                fu,
                display_name=display_name,
                description=description,
                controller_owner=controller_owner,
                research_state=research_state,
                implementation_version=implementation_version,
            )
        except KeyError as exc:
            raise FeatureNotFoundError(feature_uuid) from exc

    def update_definition(
        self,
        feature_uuid: str,
        *,
        definition_version: str,
        warmup_periods: int | None = None,
        gap_policy: str | None = None,
        memory_model: str | None = None,
        primitive_ids: list[str] | None = None,
    ) -> FeatureRecord:
        current = self.get_by_uuid(feature_uuid)
        if definition_version == current.definition_version:
            raise ValueError("definition_version must bump when updating definition")
        warm = current.warmup_periods if warmup_periods is None else int(warmup_periods)
        gap = current.gap_policy if gap_policy is None else gap_policy
        mem = current.memory_model if memory_model is None else memory_model
        prims = list(current.primitive_ids) if primitive_ids is None else list(primitive_ids)
        if not prims:
            raise ValueError("At least one primitive_id is required")
        if gap not in GAP_POLICIES:
            raise ValueError(f"Invalid gap_policy: {gap}")
        if mem not in MEMORY_MODELS:
            raise ValueError(f"Invalid memory_model: {mem}")
        for pid in prims:
            if self._primitives.get_by_id(pid) is None:
                raise ValueError(f"Unknown primitive_id: {pid}")
        dhash = compute_definition_hash(
            canonical_name=current.canonical_name,
            warmup_periods=warm,
            gap_policy=gap,
            memory_model=mem,
            primitive_ids=prims,
        )
        return self._store.replace_definition(
            current.feature_uuid,
            definition_version=definition_version,
            warmup_periods=warm,
            gap_policy=gap,
            memory_model=mem,
            definition_hash=dhash,
            primitive_ids=prims,
        )

    def list_features(
        self,
        *,
        research_state: str | None = None,
        controller_owner: str | None = None,
    ) -> list[FeatureRecord]:
        return self._store.list_all(
            research_state=research_state,
            controller_owner=controller_owner,
        )

    def find_by_primitive(self, primitive_id: str) -> list[FeatureRecord]:
        return self._store.find_by_primitive(primitive_id)

    def validate_registry(self) -> ValidationReport:
        from feature_intelligence.registry.feature_validation import validate_features

        return validate_features(self.db_path)

    def sync_from_legacy(
        self,
        data_dir: str | Path,
        *,
        mode: str = "strict",
        force: bool = False,
        research_sync: bool = False,
    ) -> SyncReport:
        from feature_intelligence.registry.feature_adapter import sync_from_legacy

        return sync_from_legacy(
            self,
            data_dir,
            mode=mode,
            force=force,
            research_sync=research_sync,
        )

    def synchronize_from_feature_registry(
        self,
        data_dir: str | Path,
        *,
        mode: str = "lenient",
        force: bool = False,
        research_sync: bool = True,
    ):
        """Import legacy catalog into FIC; returns :class:`SyncSummary`."""
        from feature_intelligence.registry.feature_registry_synchronizer import (
            synchronize_feature_registry,
        )

        return synchronize_feature_registry(
            self,
            data_dir,
            mode=mode,
            force=force,
            research_sync=research_sync,
        )
