"""Feature Registry Synchronizer — legacy catalog → FIC ``feature_registry``.

Idempotent one-way pull:
- Preserve ``FEAT_*`` when the source supplies a valid feature UUID
- Otherwise mint once; re-runs match by ``canonical_name`` and skip
- Store legacy ``FR####`` as ``legacy_feature_id``
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from feature_intelligence.registry.feature_definition_hash import compute_definition_hash
from feature_intelligence.registry.feature_ids import (
    is_valid_feature_uuid,
    normalize_feature_uuid,
)
from feature_intelligence.registry.feature_models import SyncFailure, SyncSummary
from feature_intelligence.registry.primitive_mapping import PrimitiveMappingProvider

if TYPE_CHECKING:
    from feature_intelligence.registry.feature_service import FeatureRegistryService

_IMPL_OK = frozenset({"implemented", "experimental", "deprecated", "in_progress"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _map_gap_policy(policy: dict[str, Any] | None) -> str:
    if not policy:
        return "CONTINUOUS"
    if policy.get("reset_on_gap"):
        return "RESET_ON_GAP"
    if policy.get("gap_sensitive"):
        return "GAP_AWARE"
    return "CONTINUOUS"


def _map_memory_model(policy: dict[str, Any] | None) -> str:
    lifecycle = str((policy or {}).get("lifecycle") or "tick").lower()
    mapping = {
        "tick": "TICK",
        "session": "SESSION",
        "sliding_window": "SLIDING_WINDOW",
        "day": "DAY",
    }
    return mapping.get(lifecycle, "OTHER")


def _map_research_state(status: str | None) -> str:
    s = (status or "").lower()
    return {
        "experimental": "EXPERIMENTAL",
        "implemented": "VALIDATED",
        "deprecated": "DEPRECATED",
        "in_progress": "CANDIDATE",
    }.get(s, "EXPERIMENTAL")


def extract_source_feature_uuid(feat: dict[str, Any]) -> str | None:
    """Return normalized ``FEAT_*`` if the source row carries one; else None.

    Legacy catalog uses ``FR####`` in ``feature_id`` — those are *not* FEAT ids.
    """
    for key in ("feature_uuid", "fic_feature_uuid", "feature_id"):
        raw = feat.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        if is_valid_feature_uuid(text):
            return normalize_feature_uuid(text)
    return None


def extract_legacy_feature_id(feat: dict[str, Any]) -> str | None:
    """Legacy ``FR####`` (or any non-FEAT ``feature_id``) for ``legacy_feature_id``."""
    raw = feat.get("feature_id")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if is_valid_feature_uuid(text):
        return None
    return text


def _find_existing(
    service: FeatureRegistryService,
    *,
    name: str,
    source_uuid: str | None,
):
    """Match by preserved FEAT id first, then canonical_name."""
    store = service._store  # noqa: SLF001 — synchronizer needs store lookups
    if source_uuid:
        by_uuid = store.get_by_uuid(source_uuid)
        if by_uuid is not None:
            return by_uuid
    return store.get_by_name(name)


def synchronize_feature_registry(
    service: FeatureRegistryService,
    data_dir: str | Path | None = None,
    *,
    mode: str = "lenient",
    force: bool = False,
    research_sync: bool = True,
    mapping_provider: PrimitiveMappingProvider | None = None,
    features: list[dict[str, Any]] | None = None,
) -> SyncSummary:
    """Import legacy Feature Registry catalog into FIC ``feature_registry``.

    Parameters
    ----------
    data_dir:
        Chart ``data/`` directory (``feature_registry_store.json`` etc.).
        Required unless ``features`` is provided (tests).
    mode:
        ``strict`` — unmapped primitives count as failures.
        ``lenient`` — same (reported as Failed); reserved for future soft behavior.
    force:
        When True, bump definition on hash mismatch for already-registered rows.
    research_sync:
        After new imports, run ``ResearchService.sync_research()`` so FRR shells exist.
    features:
        Optional in-memory catalog rows (skips ``build_feature_registry_catalog``).
    """
    start = time.perf_counter()
    provider_mode = mode if mode in {"strict", "lenient"} else "lenient"
    summary = SyncSummary(mode=provider_mode)
    provider = mapping_provider or PrimitiveMappingProvider()

    if features is None:
        if data_dir is None:
            summary.failures.append(
                SyncFailure(name="", reason="MISSING_DATA_DIR")
            )
            summary.failed = 1
            summary.duration_ms = int((time.perf_counter() - start) * 1000)
            summary.timestamp = _utc_now()
            return summary
        try:
            from chain_replay_ml.dataset_builder.feature_registry_catalog import (
                build_feature_registry_catalog,
            )
        except ImportError as exc:
            summary.failures.append(
                SyncFailure(name="", reason=f"LEGACY_IMPORT:{exc}")
            )
            summary.failed = 1
            summary.duration_ms = int((time.perf_counter() - start) * 1000)
            summary.timestamp = _utc_now()
            return summary
        catalog = build_feature_registry_catalog(str(data_dir))
        features = list(catalog.get("features") or [])

    summary.total_source = len(features)

    for feat in features:
        name = str(feat.get("name") or "").strip()
        legacy_id = extract_legacy_feature_id(feat)
        source_uuid = extract_source_feature_uuid(feat)

        if not name:
            summary.failed += 1
            summary.failures.append(
                SyncFailure(
                    name="",
                    reason="SKIP_EMPTY_NAME",
                    legacy_feature_id=legacy_id,
                    feature_uuid=source_uuid,
                )
            )
            continue

        if feat.get("registry_active") is False:
            summary.failed += 1
            summary.failures.append(
                SyncFailure(
                    name=name,
                    reason="SKIP_DISABLED",
                    legacy_feature_id=legacy_id,
                    feature_uuid=source_uuid,
                )
            )
            continue

        status = str(feat.get("implementation_status") or "")
        if status not in _IMPL_OK:
            summary.failed += 1
            summary.failures.append(
                SyncFailure(
                    name=name,
                    reason=f"SKIP_STATUS:{status or 'missing'}",
                    legacy_feature_id=legacy_id,
                    feature_uuid=source_uuid,
                )
            )
            continue

        primitive_ids = provider.resolve(feat)
        if not primitive_ids:
            summary.failed += 1
            summary.failures.append(
                SyncFailure(
                    name=name,
                    reason="UNMAPPED_PRIMITIVES",
                    legacy_feature_id=legacy_id,
                    feature_uuid=source_uuid,
                )
            )
            continue

        policy = feat.get("policy") if isinstance(feat.get("policy"), dict) else {}
        warmup = int(
            policy.get("effective_warmup_samples")
            or policy.get("intrinsic_warmup_samples")
            or 0
        )
        gap = _map_gap_policy(policy)
        mem = _map_memory_model(policy)
        dver = str(feat.get("feature_version") or feat.get("introduced_version") or "1.0")
        iver = str(policy.get("policy_version") or "1")
        research = _map_research_state(status)
        display = str(feat.get("display_name") or name)
        description = feat.get("description")
        owner = str(feat.get("owner") or "feature_registry_synchronizer")
        created_by = str(feat.get("created_by") or "feature_registry_synchronizer")

        new_hash = compute_definition_hash(
            canonical_name=name,
            warmup_periods=warmup,
            gap_policy=gap,
            memory_model=mem,
            primitive_ids=primitive_ids,
        )

        existing = _find_existing(service, name=name, source_uuid=source_uuid)
        if existing is not None:
            # Source FEAT + name point at different rows → conflict
            if source_uuid and existing.feature_uuid != source_uuid:
                name_row = service._store.get_by_name(name)  # noqa: SLF001
                uuid_row = service._store.get_by_uuid(source_uuid)  # noqa: SLF001
                if (
                    name_row is not None
                    and uuid_row is not None
                    and name_row.feature_uuid != uuid_row.feature_uuid
                ):
                    summary.failed += 1
                    summary.failures.append(
                        SyncFailure(
                            name=name,
                            reason="CONFLICT_FEAT_ID",
                            legacy_feature_id=legacy_id,
                            feature_uuid=source_uuid,
                        )
                    )
                    continue
                # Prefer name match when source FEAT is absent from DB
                existing = name_row or existing

            if (
                existing.legacy_feature_id
                and legacy_id
                and existing.legacy_feature_id != legacy_id
            ):
                summary.failed += 1
                summary.failures.append(
                    SyncFailure(
                        name=name,
                        reason="CONFLICT_LEGACY_ID",
                        legacy_feature_id=legacy_id,
                        feature_uuid=existing.feature_uuid,
                    )
                )
                continue

            if existing.definition_hash != new_hash and not force:
                summary.failed += 1
                summary.failures.append(
                    SyncFailure(
                        name=name,
                        reason="CONFLICT_DEFINITION",
                        legacy_feature_id=legacy_id,
                        feature_uuid=existing.feature_uuid,
                    )
                )
                continue

            if existing.definition_hash == new_hash:
                service.update_metadata(
                    existing.feature_uuid,
                    display_name=display,
                    description=None if description is None else str(description),
                    controller_owner=owner,
                    research_state=research,
                    implementation_version=iver,
                )
                summary.already_registered += 1
                summary.updated += 1
            elif force:
                service.update_definition(
                    existing.feature_uuid,
                    definition_version=dver
                    if dver != existing.definition_version
                    else f"{existing.definition_version}+sync",
                    warmup_periods=warmup,
                    gap_policy=gap,
                    memory_model=mem,
                    primitive_ids=primitive_ids,
                )
                service.update_metadata(
                    existing.feature_uuid,
                    display_name=display,
                    description=None if description is None else str(description),
                    controller_owner=owner,
                    research_state=research,
                    implementation_version=iver,
                )
                summary.already_registered += 1
                summary.updated += 1
            continue

        # New feature — preserve source FEAT_* when present
        try:
            service.register_feature(
                canonical_name=name,
                display_name=display,
                primitive_ids=primitive_ids,
                created_by=created_by,
                controller_owner=owner,
                warmup_periods=warmup,
                gap_policy=gap,
                memory_model=mem,
                definition_version=dver,
                implementation_version=iver,
                research_state=research,
                description=None if description is None else str(description),
                legacy_feature_id=legacy_id,
                transformation_uuid=None,
                feature_uuid=source_uuid,
            )
            summary.newly_imported += 1
        except Exception as exc:  # noqa: BLE001 — collect per-row sync errors
            summary.failed += 1
            summary.failures.append(
                SyncFailure(
                    name=name,
                    reason=f"REGISTER_FAIL:{exc}",
                    legacy_feature_id=legacy_id,
                    feature_uuid=source_uuid,
                )
            )

    # Fill missing FRR shells so Feature Explorer List All stays usable
    if research_sync and (summary.newly_imported > 0 or summary.already_registered > 0):
        try:
            from feature_intelligence.research.service import ResearchService

            rs = ResearchService(service.db_path)
            research_summary = rs.sync_research()
            summary.research_created = int(research_summary.created)
            summary.research_updated = int(research_summary.updated)
        except Exception as exc:  # noqa: BLE001
            summary.failures.append(
                SyncFailure(name="", reason=f"RESEARCH_SYNC_FAIL:{exc}")
            )

    summary.duration_ms = int((time.perf_counter() - start) * 1000)
    summary.timestamp = _utc_now()
    return summary
