"""Immutable inference snapshots — Layer 2 through Meta."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def _freeze_features(features: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(k): v for k, v in features.items()})


@dataclass(frozen=True)
class FeatureSnapshot:
    """Published once per grid second — never mutated after creation."""

    timestamp: float
    token: str
    features: Mapping[str, Any]
    feature_version: str
    market_state_version: str
    trading_day: str
    expiry: str
    source: str
    warmup_complete: bool
    build_sec: float
    feature_sig: str

    @classmethod
    def create(
        cls,
        *,
        timestamp: float,
        token: str,
        features: Mapping[str, Any],
        feature_version: str,
        market_state_version: str,
        trading_day: str,
        expiry: str,
        source: str,
        warmup_complete: bool,
        build_sec: float,
        feature_sig: str,
    ) -> FeatureSnapshot:
        return cls(
            timestamp=float(timestamp),
            token=str(token),
            features=_freeze_features(features),
            feature_version=str(feature_version),
            market_state_version=str(market_state_version),
            trading_day=str(trading_day),
            expiry=str(expiry),
            source=str(source),
            warmup_complete=bool(warmup_complete),
            build_sec=float(build_sec),
            feature_sig=str(feature_sig),
        )


@dataclass(frozen=True)
class PredictionResult:
    prediction: float | None
    model_id: str
    mae: float | None
    rmse: float | None
    prediction_time_ms: float
    status: str
    feature_version: str
    target: str = ""
    error: str | None = None
    tier: str = "regression"


@dataclass(frozen=True)
class PredictionSnapshot:
    """One publish per feature snapshot — map of model_id → PredictionResult."""

    timestamp: float
    token: str
    results: Mapping[str, PredictionResult]
    feature_version: str
    prediction_version: str
    models_ok: int
    models_failed: int

    @classmethod
    def create(
        cls,
        *,
        timestamp: float,
        token: str,
        results: Mapping[str, PredictionResult],
        feature_version: str,
        prediction_version: str,
    ) -> PredictionSnapshot:
        frozen = MappingProxyType(dict(results))
        ok = sum(1 for r in frozen.values() if r.status == "ok" and r.prediction is not None)
        failed = len(frozen) - ok
        return cls(
            timestamp=float(timestamp),
            token=str(token),
            results=frozen,
            feature_version=str(feature_version),
            prediction_version=str(prediction_version),
            models_ok=int(ok),
            models_failed=int(failed),
        )


@dataclass(frozen=True)
class PredictionMeta:
    """Extensible meta dict — add keys without changing Layer 4 consumers."""

    timestamp: float
    values: Mapping[str, Any]

    @classmethod
    def create(cls, *, timestamp: float, values: Mapping[str, Any]) -> PredictionMeta:
        return cls(timestamp=float(timestamp), values=MappingProxyType(dict(values)))

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)
