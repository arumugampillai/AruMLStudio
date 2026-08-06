"""Artifact Catalog types — URI identity, DAG parents, reserved capabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ArtifactType = Literal[
    "master",
    "training",
    "prediction",
    "model",
    "feature_studio",
    "diagnostics",
    "experiment",
    "eval",
    "other",
]

LifecycleStatus = Literal[
    "draft",
    "ready",
    "running",
    "completed",
    "failed",
    "superseded",
    "archived",
]

KNOWN_CAPABILITIES = frozenset(
    {
        "trainable",
        "comparable",
        "visualizable",
        "deployable",
    }
)

PipelineStepKind = Literal["dataset", "ole", "train", "eval"]


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_uri: str
    artifact_type: str
    created_at: str
    local_path: str | None = None
    parent_artifact_uris: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactRecord:
        return cls(
            artifact_uri=str(data["artifact_uri"]),
            artifact_type=str(data.get("artifact_type") or "other"),
            created_at=str(data.get("created_at") or ""),
            local_path=data.get("local_path"),
            parent_artifact_uris=list(data.get("parent_artifact_uris") or []),
            metadata=dict(data.get("metadata") or {}),
            capabilities=list(data.get("capabilities") or []),
            status=str(data.get("status") or "completed"),
        )


@dataclass(frozen=True)
class PipelineStep:
    kind: PipelineStepKind
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineStep:
        return cls(
            kind=str(data.get("kind") or "dataset"),  # type: ignore[arg-type]
            params=dict(data.get("params") or {}),
        )


@dataclass
class ExperimentContract:
    """Machine-executable experiment definition (§5.4.1)."""

    experiment_uri: str
    experiment_id: str
    actions: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    expected_benefit: str = ""
    evidence_summary: str = ""
    parent_artifact_uris: list[str] = field(default_factory=list)
    pipeline: list[PipelineStep] = field(default_factory=list)
    runnable: bool = False
    status: str = "draft"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_uri": self.experiment_uri,
            "experiment_id": self.experiment_id,
            "actions": dict(self.actions),
            "rationale": self.rationale,
            "expected_benefit": self.expected_benefit,
            "evidence_summary": self.evidence_summary,
            "parent_artifact_uris": list(self.parent_artifact_uris),
            "pipeline": [s.to_dict() for s in self.pipeline],
            "runnable": bool(self.runnable),
            "status": self.status,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentContract:
        steps = [PipelineStep.from_dict(s) for s in (data.get("pipeline") or [])]
        return cls(
            experiment_uri=str(data["experiment_uri"]),
            experiment_id=str(data["experiment_id"]),
            actions=dict(data.get("actions") or {}),
            rationale=str(data.get("rationale") or ""),
            expected_benefit=str(data.get("expected_benefit") or ""),
            evidence_summary=str(data.get("evidence_summary") or ""),
            parent_artifact_uris=list(data.get("parent_artifact_uris") or []),
            pipeline=steps,
            runnable=bool(data.get("runnable")),
            status=str(data.get("status") or "draft"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class TimelineEvent:
    artifact_uri: str
    artifact_type: str
    created_at: str
    parent_artifact_uris: list[str]
    depth: int = 0


@dataclass(frozen=True)
class ResearchMetrics:
    experiments_run: int
    best_label_strategy: str | None
    most_reused_feature_set: str | None
    avg_dataset_to_model_sec: float | None
    improvement_vs_previous: float | None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
