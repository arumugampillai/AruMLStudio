"""History-backed machine-executable experiment contracts."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from .metrics import compute_research_metrics, format_evidence_summary
from .paths import experiments_dir
from .store import ArtifactCatalogStore
from .types import ArtifactRecord, ExperimentContract, PipelineStep
from .uri import experiment_uri, is_artifact_uri


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str, *, max_len: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(text or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "exp")[:max_len]


def mint_experiment_id(prefix: str = "exp") -> str:
    token = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    import uuid

    return f"{_slug(prefix, max_len=16)}_{token}_{uuid.uuid4().hex[:6]}"


def default_pipeline_from_actions(actions: dict[str, Any]) -> list[PipelineStep]:
    """Build Dataset → OLE → Train → Eval from Planner actions."""
    steps: list[PipelineStep] = [
        PipelineStep(
            kind="dataset",
            params={
                "source_uri": actions.get("source_uri") or actions.get("dataset_uri"),
                "feature_remove": list(actions.get("feature_remove") or []),
                "feature_add": list(actions.get("feature_add") or []),
            },
        ),
        PipelineStep(
            kind="ole",
            params={
                "strategy_id": actions.get("label_strategy")
                or actions.get("strategy_id")
                or "fixed_horizon",
                "params": dict(actions.get("label_params") or actions.get("params") or {}),
            },
        ),
        PipelineStep(
            kind="train",
            params={
                "model_family": actions.get("model_family") or "xgboost",
                "feature_policy": actions.get("feature_policy"),
            },
        ),
        PipelineStep(
            kind="eval",
            params={"metrics": list(actions.get("eval_metrics") or ["mae", "accuracy"])},
        ),
    ]
    return steps


def is_runnable(contract: ExperimentContract) -> bool:
    """True when pipeline has all four step kinds and required OLE strategy."""
    kinds = {s.kind for s in contract.pipeline}
    if not {"dataset", "ole", "train", "eval"}.issubset(kinds):
        return False
    ole = next((s for s in contract.pipeline if s.kind == "ole"), None)
    if ole is None or not str(ole.params.get("strategy_id") or "").strip():
        return False
    return True


def build_contract_from_suggestion(
    store: ArtifactCatalogStore,
    suggestion: dict[str, Any],
    *,
    model_uri_parent: str | None = None,
    diagnostics_uris: list[str] | None = None,
    experiment_id: str | None = None,
) -> ExperimentContract:
    """Convert a Planner / Recommendation Engine suggestion into an ExperimentContract."""
    metrics = compute_research_metrics(store)
    evidence = format_evidence_summary(metrics)

    sid = experiment_id or str(suggestion.get("experiment_id") or "").strip()
    if not sid:
        title = str(suggestion.get("title") or suggestion.get("rule") or "planner")
        sid = mint_experiment_id(_slug(title, max_len=20))

    actions: dict[str, Any] = {}
    raw_actions = suggestion.get("actions")
    if isinstance(raw_actions, dict):
        actions.update(raw_actions)
    # Map common planner fields into actions.
    feats = suggestion.get("features") or suggestion.get("matched_features")
    if feats and "feature_remove" not in actions:
        names = []
        if isinstance(feats, list):
            for f in feats:
                if isinstance(f, dict):
                    names.append(str(f.get("name") or f.get("feature") or ""))
                else:
                    names.append(str(f))
        actions.setdefault("feature_remove", [n for n in names if n])
    if suggestion.get("label_strategy"):
        actions.setdefault("label_strategy", suggestion["label_strategy"])
    if suggestion.get("label_params"):
        actions.setdefault("label_params", dict(suggestion["label_params"]))
    # Default label strategy if still missing (history-backed preference).
    if not actions.get("label_strategy"):
        actions["label_strategy"] = (
            metrics.best_label_strategy or "fixed_horizon"
        )

    parents: list[str] = []
    for u in diagnostics_uris or []:
        if is_artifact_uri(u) and u not in parents:
            parents.append(u)
    if model_uri_parent and is_artifact_uri(model_uri_parent):
        parents.append(model_uri_parent)
    for u in suggestion.get("parent_artifact_uris") or []:
        if is_artifact_uri(str(u)) and str(u) not in parents:
            parents.append(str(u))

    rationale_parts = [
        str(suggestion.get("reason") or suggestion.get("rationale") or "").strip(),
        str(suggestion.get("hypothesis") or "").strip(),
    ]
    rationale = "\n".join(p for p in rationale_parts if p)
    expected = str(
        suggestion.get("expected_benefit")
        or suggestion.get("hypothesis")
        or "Validate recommendation against holdout."
    )

    # Enrich evidence with suggestion-local evidence if present.
    local_ev = suggestion.get("evidence")
    if isinstance(local_ev, dict) and local_ev:
        evidence = (
            evidence
            + "\n\nDiagnostics evidence:\n"
            + json.dumps(local_ev, sort_keys=True, default=str)[:2000]
        )

    pipeline = default_pipeline_from_actions(actions)
    contract = ExperimentContract(
        experiment_uri=experiment_uri(sid),
        experiment_id=sid,
        actions=actions,
        rationale=rationale,
        expected_benefit=expected,
        evidence_summary=evidence,
        parent_artifact_uris=parents,
        pipeline=pipeline,
        runnable=False,
        status="draft",
        metadata={
            "source": "planner_suggestion",
            "title": suggestion.get("title"),
            "rule": (local_ev or {}).get("rule")
            if isinstance(local_ev, dict)
            else suggestion.get("rule"),
            "created_at": _utc_now(),
        },
    )
    contract.runnable = is_runnable(contract)
    if contract.runnable:
        contract.status = "ready"
    return contract


def save_contract(data_dir: str, contract: ExperimentContract) -> str:
    """Persist contract JSON under ``artifact_catalog/experiments/``."""
    root = experiments_dir(data_dir)
    path = os.path.join(root, f"{contract.experiment_id}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(contract.to_dict(), fh, indent=2, sort_keys=True)
    return path


def load_contract(data_dir: str, experiment_id: str) -> ExperimentContract | None:
    path = os.path.join(experiments_dir(data_dir), f"{experiment_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return ExperimentContract.from_dict(json.load(fh))


def register_contract_artifact(
    store: ArtifactCatalogStore,
    contract: ExperimentContract,
    *,
    local_path: str | None = None,
) -> ArtifactRecord:
    """Register the experiment URI in the catalog."""
    return store.register(
        ArtifactRecord(
            artifact_uri=contract.experiment_uri,
            artifact_type="experiment",
            created_at=str(contract.metadata.get("created_at") or _utc_now()),
            local_path=local_path,
            parent_artifact_uris=list(contract.parent_artifact_uris),
            metadata={
                "experiment_id": contract.experiment_id,
                "actions": dict(contract.actions),
                "runnable": contract.runnable,
                "rationale": contract.rationale[:500],
                "expected_benefit": contract.expected_benefit[:500],
                "evidence_summary": contract.evidence_summary[:1000],
                "pipeline_kinds": [s.kind for s in contract.pipeline],
            },
            capabilities=["comparable"],
            status=contract.status,
        )
    )
