"""Data contracts, specifications, and Context Key for Model Taxonomy (Phase 4C.1)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .enums import (
    BASELINE_REGIME_CATALOG,
    DEFAULT_REGIME_ID,
    DEFAULT_REGIME_NAME,
    ModelLifecycleStatus,
    ModelPopulationTier,
    RegimeScope,
    TaskType,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RegimeSpec:
    """Specification of the market regime associated with a model."""

    regime_id: str = DEFAULT_REGIME_ID
    regime_name: str = DEFAULT_REGIME_NAME
    regime_version: int = 1
    regime_scope: str = RegimeScope.ALL_REGIMES.value
    definition_hash: str | None = None
    parent_regime_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime_id": str(self.regime_id),
            "regime_name": str(self.regime_name),
            "regime_version": int(self.regime_version),
            "regime_scope": str(self.regime_scope),
            "definition_hash": self.definition_hash,
            "parent_regime_id": self.parent_regime_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RegimeSpec:
        if not data or not isinstance(data, dict):
            return cls()
        rid = str(data.get("regime_id") or DEFAULT_REGIME_ID).strip()
        rname = str(data.get("regime_name") or "").strip()
        if not rname and rid in BASELINE_REGIME_CATALOG:
            rname = BASELINE_REGIME_CATALOG[rid]["name"]
        elif not rname:
            rname = DEFAULT_REGIME_NAME

        return cls(
            regime_id=rid,
            regime_name=rname,
            regime_version=int(data.get("regime_version") or 1),
            regime_scope=str(data.get("regime_scope") or (
                RegimeScope.ALL_REGIMES.value if rid == DEFAULT_REGIME_ID else RegimeScope.SPECIALIZED.value
            )),
            definition_hash=str(data["definition_hash"]) if data.get("definition_hash") else None,
            parent_regime_id=str(data["parent_regime_id"]) if data.get("parent_regime_id") else None,
        )


@dataclass(frozen=True)
class TaskSpec:
    """Specification of the mathematical prediction task."""

    task_type: TaskType
    target: str
    target_type: str = "BINARY_CLASSIFICATION"
    prediction_horizon: str = "5m"
    loss_function: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "target": str(self.target),
            "target_type": str(self.target_type),
            "prediction_horizon": str(self.prediction_horizon),
            "loss_function": self.loss_function,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, default_target: str = "label_up_5m") -> TaskSpec:
        if not data or not isinstance(data, dict):
            return cls(
                task_type=TaskType.DIRECTION_CLASSIFIER,
                target=default_target,
                target_type="BINARY_CLASSIFICATION",
                prediction_horizon="5m",
            )
        tt = TaskType.from_str(data.get("task_type") or TaskType.DIRECTION_CLASSIFIER)
        target = str(data.get("target") or default_target).strip()
        target_type = str(data.get("target_type") or (
            "CONTINUOUS" if tt.is_regression() else "BINARY_CLASSIFICATION"
        ))
        horizon = str(data.get("prediction_horizon") or "5m")
        loss = data.get("loss_function")
        return cls(
            task_type=tt,
            target=target,
            target_type=target_type,
            prediction_horizon=horizon,
            loss_function=str(loss) if loss is not None else None,
        )


@dataclass(frozen=True)
class ModelContextKey:
    """Canonical 5-tuple context key identifying a unique operational search space."""

    market: str = "NIFTY"
    sampling_interval_sec: int = 3
    task_type: TaskType = TaskType.DIRECTION_CLASSIFIER
    prediction_horizon: str = "5m"
    regime_id: str = DEFAULT_REGIME_ID

    def canonical_key_str(self) -> str:
        """Standard serialized key: e.g. 'NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001'."""
        m = str(self.market).upper().strip()
        sec = f"{int(self.sampling_interval_sec)}s"
        tt = self.task_type.value
        hor = str(self.prediction_horizon).strip()
        rid = str(self.regime_id).strip()
        return f"{m}_{sec}_{tt}_{hor}_{rid}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": str(self.market),
            "sampling_interval_sec": int(self.sampling_interval_sec),
            "task_type": self.task_type.value,
            "prediction_horizon": str(self.prediction_horizon),
            "regime_id": str(self.regime_id),
            "context_key_str": self.canonical_key_str(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelContextKey:
        return cls(
            market=str(data.get("market") or "NIFTY").upper().strip(),
            sampling_interval_sec=int(data.get("sampling_interval_sec") or 3),
            task_type=TaskType.from_str(data.get("task_type") or TaskType.DIRECTION_CLASSIFIER),
            prediction_horizon=str(data.get("prediction_horizon") or "5m").strip(),
            regime_id=str(data.get("regime_id") or DEFAULT_REGIME_ID).strip(),
        )

    @classmethod
    def from_key_str(cls, key_str: str) -> ModelContextKey:
        parts = str(key_str or "").strip().split("_")
        if len(parts) >= 5:
            market = parts[0]
            sec_str = parts[1].rstrip("s")
            try:
                sec = int(sec_str)
            except ValueError:
                sec = 3
            # In case task type has underscores (e.g. DIRECTION_CLASSIFIER)
            rid = parts[-1]
            hor = parts[-2]
            tt_str = "_".join(parts[2:-2])
            tt = TaskType.from_str(tt_str)
            return cls(
                market=market,
                sampling_interval_sec=sec,
                task_type=tt,
                prediction_horizon=hor,
                regime_id=rid,
            )
        return cls()


@dataclass
class ModelMetadata:
    """Canonical model package metadata representation."""

    model_id: str
    model_name: str
    version: int = 1
    model_family_id: str = ""
    task: TaskSpec = field(default_factory=lambda: TaskSpec(TaskType.DIRECTION_CLASSIFIER, "label_up_5m"))
    regime: RegimeSpec = field(default_factory=RegimeSpec)
    market_context: dict[str, Any] = field(default_factory=dict)
    population: ModelPopulationTier = ModelPopulationTier.EXPERIMENTAL
    status: ModelLifecycleStatus = ModelLifecycleStatus.ACTIVE
    algorithm: str = "xgboost"
    feature_count: int = 0
    lineage: dict[str, Any] = field(default_factory=dict)
    metrics_summary: dict[str, Any] = field(default_factory=dict)
    registered_at: str = field(default_factory=_utc_now_iso)

    @property
    def context_key(self) -> ModelContextKey:
        return ModelContextKey(
            market=str(self.market_context.get("market") or "NIFTY"),
            sampling_interval_sec=int(self.market_context.get("sampling_interval_sec") or 3),
            task_type=self.task.task_type,
            prediction_horizon=self.task.prediction_horizon,
            regime_id=self.regime.regime_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": str(self.model_id),
            "model_name": str(self.model_name),
            "version": int(self.version),
            "model_family_id": str(self.model_family_id),
            "task": self.task.to_dict(),
            "regime": self.regime.to_dict(),
            "market_context": dict(self.market_context),
            "population": self.population.value,
            "status": self.status.value,
            "algorithm": str(self.algorithm),
            "feature_count": int(self.feature_count),
            "lineage": dict(self.lineage),
            "metrics_summary": dict(self.metrics_summary),
            "registered_at": str(self.registered_at),
            "context_key_str": self.context_key.canonical_key_str(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        mid = str(data.get("model_id") or data.get("model_name") or "unnamed_model").strip()
        mname = str(data.get("model_name") or mid).strip()
        return cls(
            model_id=mid,
            model_name=mname,
            version=int(data.get("version") or 1),
            model_family_id=str(data.get("model_family_id") or ""),
            task=TaskSpec.from_dict(data.get("task")),
            regime=RegimeSpec.from_dict(data.get("regime")),
            market_context=dict(data.get("market_context") or {}),
            population=ModelPopulationTier.from_str(data.get("population")),
            status=ModelLifecycleStatus.from_str(data.get("status")),
            algorithm=str(data.get("algorithm") or "xgboost"),
            feature_count=int(data.get("feature_count") or 0),
            lineage=dict(data.get("lineage") or {}),
            metrics_summary=dict(data.get("metrics_summary") or {}),
            registered_at=str(data.get("registered_at") or _utc_now_iso()),
        )

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> ModelMetadata:
        return cls.from_dict(json.loads(json_str))
