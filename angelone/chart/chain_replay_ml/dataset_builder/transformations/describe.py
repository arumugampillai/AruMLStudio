"""Self-describing transformation metadata — stages, outputs, pipeline catalog.

Every transform exposes ``describe()``; the pipeline aggregates those into a
single catalog consumed by Interaction Builder, preview, validation, lineage,
ledger, docs, and APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import FeatureTransformation
from .config import normalize_transformation_config
from .registry import (
    ensure_builtin_transformations,
    get_transformation,
    list_registered_transformations,
)
from .time_shift import extract_sample_interval_from_config, normalize_sample_interval_value

MASTER_STAGE_ID = "master"
MASTER_STAGE_NAME = "Master Features"
MASTER_STAGE_ORDER = -1


@dataclass(frozen=True)
class OutputDescriptor:
    """One planned output column from a transformation stage."""

    name: str
    kind: str = ""
    source_feature: str = ""
    op: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageDescriptor:
    """Plan-time description of one pipeline stage (or Master)."""

    id: str
    name: str
    order: int
    enabled: bool = False
    depends_on: tuple[str, ...] = ()
    input_sources: tuple[str, ...] = ()
    output_descriptors: tuple[OutputDescriptor, ...] = ()
    notes: str = ""

    @property
    def output_names(self) -> list[str]:
        return [d.name for d in self.output_descriptors]


@dataclass
class PipelineDescription:
    """Full plan-time view of the transformation pipeline."""

    stages: list[StageDescriptor] = field(default_factory=list)
    master_features: list[str] = field(default_factory=list)
    sample_interval_sec: float | int | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def stage(self, stage_id: str) -> StageDescriptor | None:
        sid = str(stage_id or "").strip()
        for st in self.stages:
            if st.id == sid:
                return st
        return None

    def stages_before(self, stage_id: str) -> list[StageDescriptor]:
        """Stages with lower ``order`` than ``stage_id`` (Master included)."""
        target = self.stage(stage_id)
        if target is None:
            # Unknown target: treat as after everything registered except itself.
            try:
                ensure_builtin_transformations()
                meta = next(
                    (t for t in list_registered_transformations() if t.id == stage_id),
                    None,
                )
                cutoff = int(meta.order) if meta is not None else 10**9
            except Exception:
                cutoff = 10**9
        else:
            cutoff = int(target.order)
        return [st for st in self.stages if int(st.order) < cutoff]

    def available_before(self, stage_id: str, *, enabled_only: bool = True) -> list[str]:
        """Column names selectable by ``stage_id`` (Master + earlier outputs)."""
        names: list[str] = []
        seen: set[str] = set()
        for st in self.stages_before(stage_id):
            if enabled_only and not st.enabled and st.id != MASTER_STAGE_ID:
                continue
            for col in st.output_names:
                if col not in seen:
                    seen.add(col)
                    names.append(col)
        return names

    def columns_by_stage(
        self,
        *,
        before_stage_id: str | None = None,
        enabled_only: bool = True,
        include_empty: bool = False,
    ) -> dict[str, list[str]]:
        """Ordered map stage_id → output names (for source-aware selectors)."""
        stages = (
            self.stages_before(before_stage_id)
            if before_stage_id
            else list(self.stages)
        )
        out: dict[str, list[str]] = {}
        for st in stages:
            if enabled_only and not st.enabled and st.id != MASTER_STAGE_ID:
                continue
            cols = list(st.output_names)
            if cols or include_empty or st.id == MASTER_STAGE_ID:
                out[st.id] = cols
        return out

    def source_choices(
        self,
        *,
        before_stage_id: str,
        enabled_only: bool = True,
        require_outputs: bool = True,
    ) -> list[tuple[str, str]]:
        """``(stage_id, display_name)`` for UI source dropdowns."""
        choices: list[tuple[str, str]] = []
        for st in self.stages_before(before_stage_id):
            if enabled_only and not st.enabled and st.id != MASTER_STAGE_ID:
                continue
            if require_outputs and st.id != MASTER_STAGE_ID and not st.output_names:
                continue
            choices.append((st.id, st.name))
        return choices

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_interval_sec": self.sample_interval_sec,
            "master_features": list(self.master_features),
            "stages": [
                {
                    "id": st.id,
                    "name": st.name,
                    "order": st.order,
                    "enabled": st.enabled,
                    "depends_on": list(st.depends_on),
                    "input_sources": list(st.input_sources),
                    "outputs": [d.name for d in st.output_descriptors],
                    "output_descriptors": [
                        {
                            "name": d.name,
                            "kind": d.kind,
                            "source_feature": d.source_feature,
                            "op": d.op,
                            "meta": dict(d.meta),
                        }
                        for d in st.output_descriptors
                    ],
                    "notes": st.notes,
                }
                for st in self.stages
            ],
        }


def master_stage_descriptor(master_features: list[str] | None = None) -> StageDescriptor:
    feats = [str(f).strip() for f in (master_features or []) if str(f).strip()]
    return StageDescriptor(
        id=MASTER_STAGE_ID,
        name=MASTER_STAGE_NAME,
        order=MASTER_STAGE_ORDER,
        enabled=True,
        depends_on=(),
        input_sources=(),
        output_descriptors=tuple(OutputDescriptor(name=n, kind="master") for n in feats),
        notes="Canonical Master / Feature Registry columns.",
    )


def describe_pipeline_stages(
    config: dict[str, Any] | list[Any] | None = None,
    *,
    master_features: list[str] | None = None,
    sample_interval_sec: float | int | None = None,
    include_disabled: bool = True,
) -> PipelineDescription:
    """Build the self-describing pipeline catalog from config + registry.

    Walks transforms in execution order. Each stage's ``describe()`` sees the
    accumulating upstream catalog so planned outputs can reference prior stages.
    """
    ensure_builtin_transformations()
    cfg = normalize_transformation_config(config)
    interval = normalize_sample_interval_value(sample_interval_sec)
    if interval is None:
        interval = extract_sample_interval_from_config(cfg)

    master = [str(f).strip() for f in (master_features or []) if str(f).strip()]
    stages: list[StageDescriptor] = [master_stage_descriptor(master)]

    entry_by_id: dict[str, dict[str, Any]] = {}
    for entry in list(cfg.get("transformations") or []):
        if not isinstance(entry, dict):
            continue
        tid = str(entry.get("id") or "").strip()
        if tid:
            entry_by_id[tid] = entry

    # All registered transforms (stable order); disabled stages still describe.
    registered = sorted(
        list_registered_transformations(),
        key=lambda t: (int(t.order), str(t.id)),
    )
    for meta in registered:
        entry = entry_by_id.get(meta.id) or {}
        enabled = bool(entry.get("enabled", False))
        if not include_disabled and not enabled:
            continue
        params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
        order = int(entry.get("order", meta.order) or meta.order)
        try:
            inst = get_transformation(meta.id)
        except KeyError:
            continue
        inst.enabled = enabled
        inst.order = order
        if isinstance(entry.get("depends_on"), (list, tuple)) and entry.get("depends_on"):
            inst.depends_on = [
                str(d).strip() for d in entry["depends_on"] if str(d).strip()
            ]
        else:
            inst.depends_on = list(getattr(type(inst), "depends_on", []) or [])
        setattr(inst, "params", dict(params or {}))

        upstream = PipelineDescription(
            stages=list(stages),
            master_features=list(master),
            sample_interval_sec=interval,
            config=cfg,
        )
        try:
            stage = inst.describe(
                dict(params or {}),
                upstream=upstream,
                master_features=master,
                sample_interval_sec=interval,
                enabled=enabled,
            )
        except Exception:
            stage = StageDescriptor(
                id=str(inst.id),
                name=str(inst.name or inst.id),
                order=int(inst.order),
                enabled=enabled,
                depends_on=tuple(inst.depends_on or ()),
                input_sources=(MASTER_STAGE_ID,),
                output_descriptors=(),
                notes="describe() failed; outputs unavailable.",
            )
        # Prefer config order if describe omitted it.
        if stage.order != order:
            stage = StageDescriptor(
                id=stage.id,
                name=stage.name,
                order=order,
                enabled=stage.enabled,
                depends_on=stage.depends_on,
                input_sources=stage.input_sources,
                output_descriptors=stage.output_descriptors,
                notes=stage.notes,
            )
        stages.append(stage)

    return PipelineDescription(
        stages=stages,
        master_features=master,
        sample_interval_sec=interval,
        config=cfg,
    )


def make_stage_descriptor(
    transform: FeatureTransformation,
    *,
    enabled: bool,
    outputs: list[OutputDescriptor] | tuple[OutputDescriptor, ...],
    input_sources: list[str] | tuple[str, ...] | None = None,
    notes: str = "",
) -> StageDescriptor:
    """Helper for transform ``describe()`` implementations."""
    src = tuple(input_sources) if input_sources is not None else (MASTER_STAGE_ID,)
    return StageDescriptor(
        id=str(transform.id),
        name=str(transform.name or transform.id),
        order=int(transform.order),
        enabled=bool(enabled),
        depends_on=tuple(str(d) for d in (transform.depends_on or [])),
        input_sources=src,
        output_descriptors=tuple(outputs),
        notes=str(notes or ""),
    )


__all__ = [
    "MASTER_STAGE_ID",
    "MASTER_STAGE_NAME",
    "MASTER_STAGE_ORDER",
    "OutputDescriptor",
    "StageDescriptor",
    "PipelineDescription",
    "describe_pipeline_stages",
    "make_stage_descriptor",
    "master_stage_descriptor",
]
