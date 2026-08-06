"""Execute runnable experiment contracts — Dataset → OLE → Train → Eval.

Orchestrates existing engines via injectable step handlers. Default handlers
perform a dry-run that still registers result URIs in the Catalog (tests /
Planner "Run" path without full training). Real engines can be wired by
passing ``step_handlers``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .experiment_contracts import is_runnable, load_contract, save_contract
from .paths import experiments_dir
from .store import ArtifactCatalogError, ArtifactCatalogStore
from .types import ArtifactRecord, ExperimentContract, PipelineStep
from .uri import eval_uri, model_uri, training_uri


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


StepHandler = Callable[
    [PipelineStep, "ExperimentRunContext"],
    "StepResult",
]


@dataclass
class StepResult:
    ok: bool
    artifact_uri: str | None = None
    local_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ExperimentRunContext:
    data_dir: str
    store: ArtifactCatalogStore
    contract: ExperimentContract
    produced_uris: list[str] = field(default_factory=list)
    step_outputs: dict[str, StepResult] = field(default_factory=dict)
    dry_run: bool = True


@dataclass
class ExperimentRunResult:
    ok: bool
    experiment_uri: str
    status: str
    produced_uris: list[str] = field(default_factory=list)
    step_outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    improvement_pct: float | None = None


def _dry_dataset(step: PipelineStep, ctx: ExperimentRunContext) -> StepResult:
    src = step.params.get("source_uri")
    meta = {
        "feature_remove": step.params.get("feature_remove") or [],
        "feature_add": step.params.get("feature_add") or [],
        "source_uri": src,
        "dry_run": True,
    }
    # Reuse source URI as dataset pointer when present; else mint under training family as selection.
    uri = str(src) if src else training_uri(f"dataset_sel_{ctx.contract.experiment_id}")
    return StepResult(ok=True, artifact_uri=uri, metadata=meta)


def _dry_ole(step: PipelineStep, ctx: ExperimentRunContext) -> StepResult:
    strategy = str(step.params.get("strategy_id") or "fixed_horizon")
    artifact_id = f"training_dataset_{strategy}_{ctx.contract.experiment_id}"
    uri = training_uri(artifact_id)
    out_dir = os.path.join(experiments_dir(ctx.data_dir), ctx.contract.experiment_id, "ole")
    os.makedirs(out_dir, exist_ok=True)
    run_meta = {
        "strategy": strategy,
        "params": dict(step.params.get("params") or {}),
        "experiment_uri": ctx.contract.experiment_uri,
        "dry_run": True,
        "created_at_utc": _utc_now(),
        "rows": 0,
    }
    meta_path = os.path.join(out_dir, "run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(run_meta, fh, indent=2)
    parents = list(ctx.produced_uris[-1:]) if ctx.produced_uris else list(
        ctx.contract.parent_artifact_uris
    )
    ctx.store.register(
        ArtifactRecord(
            artifact_uri=uri,
            artifact_type="training",
            created_at=_utc_now(),
            local_path=out_dir,
            parent_artifact_uris=parents + [ctx.contract.experiment_uri],
            metadata={"strategy": strategy, "dry_run": True, "rows": 0},
            capabilities=["trainable", "comparable"],
            status="completed",
        )
    )
    return StepResult(ok=True, artifact_uri=uri, local_path=out_dir, metadata=run_meta)


def _dry_train(step: PipelineStep, ctx: ExperimentRunContext) -> StepResult:
    name = f"{ctx.contract.experiment_id}_model"
    uri = model_uri(name)
    out_dir = os.path.join(experiments_dir(ctx.data_dir), ctx.contract.experiment_id, "model")
    os.makedirs(out_dir, exist_ok=True)
    parents = [u for u in ctx.produced_uris if u.startswith("aruneo://training/")]
    if not parents and ctx.produced_uris:
        parents = [ctx.produced_uris[-1]]
    parents = parents + [ctx.contract.experiment_uri]
    ctx.store.register(
        ArtifactRecord(
            artifact_uri=uri,
            artifact_type="model",
            created_at=_utc_now(),
            local_path=out_dir,
            parent_artifact_uris=parents,
            metadata={
                "model_family": step.params.get("model_family"),
                "dry_run": True,
                "experiment_id": ctx.contract.experiment_id,
            },
            capabilities=["comparable", "deployable", "visualizable"],
            status="completed",
        )
    )
    return StepResult(ok=True, artifact_uri=uri, local_path=out_dir, metadata={"dry_run": True})


def _dry_eval(step: PipelineStep, ctx: ExperimentRunContext) -> StepResult:
    uri = eval_uri(ctx.contract.experiment_id)
    # Synthetic improvement for metrics demos when not provided.
    improvement = float(ctx.contract.metadata.get("improvement_pct") or 0.0)
    out_dir = os.path.join(experiments_dir(ctx.data_dir), ctx.contract.experiment_id, "eval")
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "metrics": step.params.get("metrics") or [],
        "improvement_pct": improvement,
        "dry_run": True,
        "created_at_utc": _utc_now(),
    }
    path = os.path.join(out_dir, "result.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    parents = list(ctx.produced_uris[-1:]) + [ctx.contract.experiment_uri]
    ctx.store.register(
        ArtifactRecord(
            artifact_uri=uri,
            artifact_type="eval",
            created_at=_utc_now(),
            local_path=out_dir,
            parent_artifact_uris=parents,
            metadata=result,
            capabilities=["comparable", "visualizable"],
            status="completed",
        )
    )
    return StepResult(ok=True, artifact_uri=uri, local_path=out_dir, metadata=result)


DEFAULT_DRY_HANDLERS: dict[str, StepHandler] = {
    "dataset": _dry_dataset,
    "ole": _dry_ole,
    "train": _dry_train,
    "eval": _dry_eval,
}


class ExperimentExecutor:
    """Run a contract pipeline and register results in the Catalog."""

    def __init__(
        self,
        data_dir: str,
        store: ArtifactCatalogStore,
        *,
        step_handlers: dict[str, StepHandler] | None = None,
        dry_run: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.store = store
        self.dry_run = dry_run
        self.handlers = dict(DEFAULT_DRY_HANDLERS)
        if step_handlers:
            self.handlers.update(step_handlers)

    def run(self, contract: ExperimentContract) -> ExperimentRunResult:
        if not is_runnable(contract) and not contract.runnable:
            # Re-check in case flag stale.
            if not is_runnable(contract):
                return ExperimentRunResult(
                    ok=False,
                    experiment_uri=contract.experiment_uri,
                    status="draft",
                    error="contract is not runnable (incomplete pipeline)",
                )

        # Mark running
        contract.status = "running"
        save_contract(self.data_dir, contract)
        try:
            self.store.update_status(contract.experiment_uri, "running")
        except ArtifactCatalogError:
            # Register if missing.
            from .experiment_contracts import register_contract_artifact

            register_contract_artifact(
                self.store,
                contract,
                local_path=os.path.join(
                    experiments_dir(self.data_dir), f"{contract.experiment_id}.json"
                ),
            )
            self.store.update_status(contract.experiment_uri, "running")

        ctx = ExperimentRunContext(
            data_dir=self.data_dir,
            store=self.store,
            contract=contract,
            dry_run=self.dry_run,
        )
        step_meta: dict[str, Any] = {}
        try:
            for step in contract.pipeline:
                handler = self.handlers.get(step.kind)
                if handler is None:
                    raise ArtifactCatalogError(f"no handler for pipeline step: {step.kind}")
                result = handler(step, ctx)
                step_meta[step.kind] = {
                    "ok": result.ok,
                    "artifact_uri": result.artifact_uri,
                    "error": result.error,
                    "metadata": result.metadata,
                }
                if not result.ok:
                    raise ArtifactCatalogError(
                        result.error or f"step {step.kind} failed"
                    )
                if result.artifact_uri:
                    ctx.produced_uris.append(result.artifact_uri)
                ctx.step_outputs[step.kind] = result

            improvement = None
            eval_out = ctx.step_outputs.get("eval")
            if eval_out and isinstance(eval_out.metadata, dict):
                try:
                    improvement = float(eval_out.metadata.get("improvement_pct"))
                except (TypeError, ValueError):
                    improvement = None

            # Complete experiment artifact with results.
            existing = self.store.get(contract.experiment_uri)
            meta = dict(existing.metadata) if existing else {}
            meta["result"] = {
                "produced_uris": list(ctx.produced_uris),
                "improvement_pct": improvement,
                "dry_run": self.dry_run,
                "completed_at": _utc_now(),
            }
            if improvement is not None:
                meta["improvement_pct"] = improvement
            self.store.register(
                ArtifactRecord(
                    artifact_uri=contract.experiment_uri,
                    artifact_type="experiment",
                    created_at=(existing.created_at if existing else _utc_now()),
                    local_path=existing.local_path if existing else None,
                    parent_artifact_uris=list(contract.parent_artifact_uris),
                    metadata=meta,
                    capabilities=list(existing.capabilities) if existing else ["comparable"],
                    status="completed",
                )
            )
            contract.status = "completed"
            contract.metadata = {**contract.metadata, **meta}
            save_contract(self.data_dir, contract)
            return ExperimentRunResult(
                ok=True,
                experiment_uri=contract.experiment_uri,
                status="completed",
                produced_uris=list(ctx.produced_uris),
                step_outputs=step_meta,
                improvement_pct=improvement,
            )
        except Exception as exc:  # noqa: BLE001 — surface to Planner UI
            contract.status = "failed"
            contract.metadata = {
                **contract.metadata,
                "error": str(exc),
                "failed_at": _utc_now(),
            }
            save_contract(self.data_dir, contract)
            try:
                self.store.update_status(contract.experiment_uri, "failed")
            except ArtifactCatalogError:
                pass
            return ExperimentRunResult(
                ok=False,
                experiment_uri=contract.experiment_uri,
                status="failed",
                produced_uris=list(ctx.produced_uris),
                step_outputs=step_meta,
                error=str(exc),
            )


def run_experiment(
    data_dir: str,
    store: ArtifactCatalogStore,
    contract: ExperimentContract,
    *,
    dry_run: bool = True,
    step_handlers: dict[str, StepHandler] | None = None,
) -> ExperimentRunResult:
    return ExperimentExecutor(
        data_dir,
        store,
        step_handlers=step_handlers,
        dry_run=dry_run,
    ).run(contract)


def run_experiment_by_id(
    data_dir: str,
    store: ArtifactCatalogStore,
    experiment_id: str,
    *,
    dry_run: bool = True,
) -> ExperimentRunResult:
    contract = load_contract(data_dir, experiment_id)
    if contract is None:
        return ExperimentRunResult(
            ok=False,
            experiment_uri=f"aruneo://experiment/{experiment_id}",
            status="failed",
            error=f"contract not found: {experiment_id}",
        )
    return run_experiment(data_dir, store, contract, dry_run=dry_run)
