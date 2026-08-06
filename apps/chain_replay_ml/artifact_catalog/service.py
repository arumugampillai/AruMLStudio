"""Public Artifact Catalog / Phase 7 research capability service."""

from __future__ import annotations

from typing import Any

from .experiment_contracts import (
    build_contract_from_suggestion,
    is_runnable,
    load_contract,
    register_contract_artifact,
    save_contract,
)
from .executor import ExperimentRunResult, run_experiment, run_experiment_by_id
from .indexer import rebuild_catalog_index
from .metrics import compute_research_metrics, format_evidence_summary
from .store import ArtifactCatalogStore
from .timeline import lineage_ancestors, lineage_chain_uris, timeline_chronological
from .types import ArtifactRecord, ExperimentContract, ResearchMetrics, TimelineEvent


class ArtifactCatalogService:
    """Facade over Catalog store, indexer, timeline, metrics, and contracts."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.store = ArtifactCatalogStore(data_dir)

    def __enter__(self) -> ArtifactCatalogService:
        self.store.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.store.close()

    def open(self) -> None:
        self.store.open()

    def close(self) -> None:
        self.store.close()

    def register(self, record: ArtifactRecord) -> ArtifactRecord:
        return self.store.register(record)

    def get(self, artifact_uri: str) -> ArtifactRecord | None:
        return self.store.get(artifact_uri)

    def list_all(self) -> list[ArtifactRecord]:
        return self.store.list_all()

    def rebuild_index(self, *, ole_training_root: str | None = None) -> dict[str, Any]:
        return rebuild_catalog_index(
            self.store, self.data_dir, ole_training_root=ole_training_root
        )

    def timeline(self) -> list[TimelineEvent]:
        return timeline_chronological(self.store)

    def lineage(self, artifact_uri: str) -> list[TimelineEvent]:
        return lineage_ancestors(self.store, artifact_uri)

    def lineage_chain(self, artifact_uri: str) -> list[str]:
        return lineage_chain_uris(self.store, artifact_uri)

    def metrics(self) -> ResearchMetrics:
        return compute_research_metrics(self.store)

    def evidence_summary(self) -> str:
        return format_evidence_summary(self.metrics())

    def create_contract_from_suggestion(
        self,
        suggestion: dict[str, Any],
        *,
        model_uri_parent: str | None = None,
        diagnostics_uris: list[str] | None = None,
        persist: bool = True,
    ) -> ExperimentContract:
        contract = build_contract_from_suggestion(
            self.store,
            suggestion,
            model_uri_parent=model_uri_parent,
            diagnostics_uris=diagnostics_uris,
        )
        path = None
        if persist:
            path = save_contract(self.data_dir, contract)
        register_contract_artifact(self.store, contract, local_path=path)
        return contract

    def run_contract(
        self,
        contract: ExperimentContract,
        *,
        dry_run: bool = True,
    ) -> ExperimentRunResult:
        if not is_runnable(contract):
            contract.runnable = is_runnable(contract)
        return run_experiment(self.data_dir, self.store, contract, dry_run=dry_run)

    def run_by_id(
        self, experiment_id: str, *, dry_run: bool = True
    ) -> ExperimentRunResult:
        return run_experiment_by_id(
            self.data_dir, self.store, experiment_id, dry_run=dry_run
        )

    def load_contract(self, experiment_id: str) -> ExperimentContract | None:
        return load_contract(self.data_dir, experiment_id)
