"""Strategy Registry service — create, clone, version, champion."""

from __future__ import annotations

import json
import os
from typing import Any

from .hashes import strategy_config_hash
from .paths import strategy_package_dir
from .schema import (
    LIFECYCLE_LABELS,
    default_strategy_template,
    normalize_strategy_config,
    safe_strategy_slug,
    validate_strategy_config,
)
from .store import StrategyRegistryStore


def _version_label(num: int) -> str:
    return f"v{num}"


def _write_version_package(data_dir: str, version: dict[str, Any]) -> str:
    strategy_id = str(version["strategy_id"])
    label = str(version["version_label"])
    pkg = strategy_package_dir(data_dir, strategy_id, label)
    os.makedirs(pkg, exist_ok=True)
    doc = {
        "version_id": version.get("version_id"),
        "strategy_id": strategy_id,
        "version_label": label,
        "version_number": version.get("version_number"),
        "config_hash": version.get("config_hash"),
        "lifecycle": version.get("lifecycle"),
        "display_name": version.get("display_name"),
        "description": version.get("description"),
        "config": version.get("config"),
        "created_on": version.get("created_on"),
    }
    with open(os.path.join(pkg, "strategy.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, default=str)
    return pkg


def create_strategy(
    data_dir: str,
    *,
    display_name: str,
    description: str | None = None,
    config: dict[str, Any] | None = None,
    lifecycle: str = "new_strategy",
) -> dict[str, Any]:
    cfg = normalize_strategy_config(config or default_strategy_template(name=display_name))
    cfg["name"] = display_name
    if description is not None:
        cfg["description"] = description
    errors = validate_strategy_config(cfg)
    if errors:
        raise ValueError("; ".join(errors))

    cfg_hash = strategy_config_hash(cfg)
    slug = safe_strategy_slug(display_name)

    with StrategyRegistryStore(data_dir) as store:
        if store.get_profile_by_slug(slug):
            slug = f"{slug}_{os.urandom(3).hex()}"

        profile = store.insert_profile({
            "display_name": display_name,
            "description": cfg.get("description") or description,
            "slug": slug,
            "status": "active",
        })
        strategy_id = profile["strategy_id"]
        version = store.insert_version({
            "strategy_id": strategy_id,
            "version_label": _version_label(1),
            "version_number": 1,
            "lifecycle": lifecycle,
            "config_hash": cfg_hash,
            "display_name": display_name,
            "description": cfg.get("description"),
            "config": cfg,
        })
        store.update_profile_champion(
            strategy_id,
            version_id=version["version_id"],
            version_label=version["version_label"],
            version_number=1,
            config_hash=cfg_hash,
            display_name=display_name,
            description=cfg.get("description"),
        )

    _write_version_package(data_dir, version)
    return get_strategy_detail(data_dir, strategy_id) or {}


def create_strategy_version(
    data_dir: str,
    *,
    strategy_id: str,
    config: dict[str, Any],
    lifecycle: str = "edit",
    parent_version_id: str | None = None,
    set_champion: bool = True,
) -> dict[str, Any]:
    cfg = normalize_strategy_config(config)
    errors = validate_strategy_config(cfg)
    if errors:
        raise ValueError("; ".join(errors))
    cfg_hash = strategy_config_hash(cfg)

    with StrategyRegistryStore(data_dir) as store:
        profile = store.get_profile(strategy_id)
        if not profile:
            raise ValueError(f"strategy not found: {strategy_id}")

        existing = store.find_version_by_hash(strategy_id, cfg_hash)
        if existing and lifecycle != "clone":
            if set_champion:
                store.update_profile_champion(
                    strategy_id,
                    version_id=existing["version_id"],
                    version_label=existing["version_label"],
                    version_number=int(existing["version_number"]),
                    config_hash=cfg_hash,
                )
            return get_strategy_version(data_dir, existing["version_id"]) or existing

        vnum = store.next_version_number(strategy_id)
        version = store.insert_version({
            "strategy_id": strategy_id,
            "version_label": _version_label(vnum),
            "version_number": vnum,
            "parent_version_id": parent_version_id,
            "lifecycle": lifecycle,
            "config_hash": cfg_hash,
            "display_name": cfg.get("name") or profile.get("display_name"),
            "description": cfg.get("description") or profile.get("description"),
            "config": cfg,
        })
        if set_champion:
            store.update_profile_champion(
                strategy_id,
                version_id=version["version_id"],
                version_label=version["version_label"],
                version_number=vnum,
                config_hash=cfg_hash,
                display_name=version.get("display_name"),
                description=version.get("description"),
            )

    _write_version_package(data_dir, version)
    return get_strategy_version(data_dir, version["version_id"]) or version


def clone_strategy_version(
    data_dir: str,
    *,
    source_version_id: str,
    display_name: str | None = None,
    description: str | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = get_strategy_version(data_dir, source_version_id)
    if not source:
        raise ValueError(f"version not found: {source_version_id}")

    cfg = normalize_strategy_config(source.get("config") or {})
    if config_overrides:
        for section, values in config_overrides.items():
            if section in ("name", "description"):
                cfg[section] = values
            elif isinstance(values, dict) and isinstance(cfg.get(section), dict):
                cfg[section].update(values)
    if display_name:
        cfg["name"] = display_name
    if description is not None:
        cfg["description"] = description

    with StrategyRegistryStore(data_dir) as store:
        profile = store.get_profile(source["strategy_id"])
        if not profile:
            raise ValueError("source strategy profile missing")

    if display_name and display_name != profile.get("display_name"):
        detail = create_strategy(
            data_dir,
            display_name=display_name,
            description=description or cfg.get("description"),
            config=cfg,
            lifecycle="clone",
        )
        return detail.get("champion_version") or detail

    return create_strategy_version(
        data_dir,
        strategy_id=source["strategy_id"],
        config=cfg,
        lifecycle="clone",
        parent_version_id=source_version_id,
        set_champion=True,
    )


def set_champion_version(data_dir: str, strategy_id: str, version_id: str) -> dict[str, Any]:
    with StrategyRegistryStore(data_dir) as store:
        profile = store.get_profile(strategy_id)
        version = store.get_version(version_id)
        if not profile:
            raise ValueError(f"strategy not found: {strategy_id}")
        if not version or version["strategy_id"] != strategy_id:
            raise ValueError(f"version not found for strategy: {version_id}")
        store.update_profile_champion(
            strategy_id,
            version_id=version_id,
            version_label=version["version_label"],
            version_number=int(version["version_number"]),
            config_hash=version["config_hash"],
            display_name=version.get("display_name") or profile.get("display_name"),
            description=version.get("description") or profile.get("description"),
        )
    return get_strategy_detail(data_dir, strategy_id) or {}


def archive_strategy(data_dir: str, strategy_id: str) -> dict[str, Any]:
    with StrategyRegistryStore(data_dir) as store:
        profile = store.get_profile(strategy_id)
        if not profile:
            raise ValueError(f"strategy not found: {strategy_id}")
        store.set_profile_status(strategy_id, "archived")
    return get_strategy_detail(data_dir, strategy_id) or {}


def list_strategies(data_dir: str, *, limit: int = 100, include_archived: bool = False) -> list[dict[str, Any]]:
    with StrategyRegistryStore(data_dir) as store:
        rows = store.list_profiles(limit=limit, include_archived=include_archived)
    for row in rows:
        row["is_champion"] = True
    return rows


def get_strategy_version(data_dir: str, version_id: str) -> dict[str, Any] | None:
    with StrategyRegistryStore(data_dir) as store:
        version = store.get_version(version_id)
        if not version:
            return None
        profile = store.get_profile(version["strategy_id"])
        version["strategy_display_name"] = profile.get("display_name") if profile else None
        version["is_champion"] = bool(
            profile and profile.get("current_version_id") == version_id
        )
        version["lifecycle_label"] = LIFECYCLE_LABELS.get(
            version.get("lifecycle") or "", version.get("lifecycle")
        )
        return version


def get_strategy_detail(data_dir: str, strategy_id: str) -> dict[str, Any] | None:
    with StrategyRegistryStore(data_dir) as store:
        profile = store.get_profile(strategy_id)
        if not profile:
            return None
        versions = store.list_versions(strategy_id)
        champion_id = profile.get("current_version_id")
        for v in versions:
            v["is_champion"] = v.get("version_id") == champion_id
            v["lifecycle_label"] = LIFECYCLE_LABELS.get(v.get("lifecycle") or "", v.get("lifecycle"))
        champion = store.get_version(champion_id) if champion_id else None
        return {
            "ok": True,
            "profile": profile,
            "champion_version": champion,
            "versions": versions,
            "version_count": len(versions),
        }


def compare_strategy_versions(data_dir: str, version_a: str, version_b: str) -> dict[str, Any]:
    a = get_strategy_version(data_dir, version_a)
    b = get_strategy_version(data_dir, version_b)
    if not a or not b:
        return {"ok": False, "error": "one or both versions not found"}

    cfg_a = normalize_strategy_config(a.get("config") or {})
    cfg_b = normalize_strategy_config(b.get("config") or {})
    changes: list[dict[str, Any]] = []

    def _walk(path: str, left: Any, right: Any) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            keys = sorted(set(left) | set(right))
            for k in keys:
                _walk(f"{path}.{k}" if path else k, left.get(k), right.get(k))
            return
        if left != right:
            changes.append({"path": path, "a": left, "b": right})

    for section in ("entry", "exit", "stop", "target", "hold_time", "confidence", "position_size", "execution"):
        _walk(section, cfg_a.get(section), cfg_b.get(section))

    return {
        "ok": True,
        "version_a": a,
        "version_b": b,
        "same_hash": a.get("config_hash") == b.get("config_hash"),
        "changes": changes,
    }


def get_default_template() -> dict[str, Any]:
    return default_strategy_template()
