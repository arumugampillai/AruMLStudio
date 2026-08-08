"""Default build config for index-page master dataset adds."""

from __future__ import annotations

from typing import Any

from .feature_plugins import resolve_implemented_features_for_selection
from .master_naming import master_dataset_slug, resolve_master_db_path
from .orchestrator import DatasetBuildConfig, _load_feature_registry


def default_master_column_counts(registry: dict[str, Any] | None = None) -> tuple[int, int]:
    reg = registry or _load_feature_registry()
    feature_selection = default_master_feature_selection(reg)
    _, implemented, _, _ = resolve_implemented_features_for_selection(feature_selection, reg)
    targets = default_master_prediction_targets()
    return len(implemented), len(targets.get("horizonsSec") or [])


def default_master_feature_selection(registry: dict[str, Any] | None = None) -> dict[str, Any]:
    reg = registry or _load_feature_registry()
    groups = list(reg.get("groupOrder") or [])
    features: list[str] = []
    for gid in groups:
        feats = (reg.get("groups") or {}).get(gid, {}).get("features") or []
        features.extend(str(f) for f in feats)
    features = list(dict.fromkeys(features))
    return {
        "profile": "default",
        "enabledGroups": groups,
        "enabledFeatures": features,
        "applied": True,
    }


def default_master_sampling(interval_sec: int = 10, *, sliding_stride_sec: int | None = None) -> dict[str, Any]:
    window = int(interval_sec)
    stride = int(sliding_stride_sec) if sliding_stride_sec is not None else window
    return {
        "configVersion": 1,
        "trainingIntervalSec": window,
        "slidingStrideSec": stride,
        "samplingMethod": "fixed_interval",
        "applied": True,
    }


def default_master_strike_selection() -> dict[str, Any]:
    return {
        "configVersion": 1,
        "mode": "atm_band",
        "atmBand": 10,
        "premiumMin": 15,
        "premiumMax": 30,
        "premiumIgnoreOutside": False,
        "deltaType": "absolute",
        "deltaMin": 0.15,
        "deltaMax": 0.50,
        "customOffsets": [-3, -2, -1, 0, 1, 2, 3],
        "applied": True,
    }


def default_master_prediction_targets() -> dict[str, Any]:
    return {
        "configVersion": 1,
        "targetType": "future_ltp",
        "horizonsSec": [10, 30, 60, 300],
        "applied": True,
    }


def build_master_dataset_config(
    *,
    data_dir: str,
    market: str,
    interval_sec: int,
    sources: list[dict[str, Any]],
    master_db_path: str | None = None,
) -> DatasetBuildConfig:
    registry = _load_feature_registry()
    slug = master_dataset_slug(market=market, sampling_interval_sec=interval_sec)
    path = master_db_path or resolve_master_db_path(
        data_dir,
        market=market,
        sampling_interval_sec=interval_sec,
    )
    return DatasetBuildConfig(
        dataset_name=slug,
        sources=list(sources),
        sampling=default_master_sampling(interval_sec),
        strike_selection=default_master_strike_selection(),
        prediction_targets=default_master_prediction_targets(),
        feature_selection=default_master_feature_selection(registry),
        feature_registry=registry,
        data_dir=data_dir,
        build_mode="append",
        storage_backend="master_sqlite",
        master_db_path=path,
    )
