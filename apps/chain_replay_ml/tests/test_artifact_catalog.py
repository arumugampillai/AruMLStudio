"""Phase 7 Artifact Catalog / Timeline / Metrics / Experiment Contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chain_replay_ml.artifact_catalog import (
    ArtifactCatalogError,
    ArtifactCatalogService,
    ArtifactCatalogStore,
    ArtifactRecord,
    ArtifactUriError,
    compute_research_metrics,
    is_artifact_uri,
    is_runnable,
    lineage_chain_uris,
    mint_uri,
    model_uri,
    parse_uri,
    rebuild_catalog_index,
    training_uri,
)
from chain_replay_ml.artifact_catalog.uri import experiment_uri


class UriTests(unittest.TestCase):
    def test_mint_parse_roundtrip(self) -> None:
        uri = mint_uri("model", "xgb_v1")
        self.assertEqual(uri, "aruneo://model/xgb_v1")
        fam, segs = parse_uri(uri)
        self.assertEqual(fam, "model")
        self.assertEqual(segs, ["xgb_v1"])

    def test_invalid_uri(self) -> None:
        self.assertFalse(is_artifact_uri("http://x"))
        with self.assertRaises(ArtifactUriError):
            parse_uri("aruneo://onlyfamily")


class StoreTests(unittest.TestCase):
    def test_register_get_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with ArtifactCatalogStore(tmp) as store:
                a = store.register(
                    ArtifactRecord(
                        artifact_uri=training_uri("t1"),
                        artifact_type="training",
                        created_at="2026-01-01T00:00:00+00:00",
                        parent_artifact_uris=[],
                        metadata={"strategy": "fixed_horizon"},
                    )
                )
                b = store.register(
                    ArtifactRecord(
                        artifact_uri=model_uri("m1"),
                        artifact_type="model",
                        created_at="2026-01-02T00:00:00+00:00",
                        parent_artifact_uris=[a.artifact_uri],
                        capabilities=["deployable"],
                    )
                )
                self.assertEqual(store.get(b.artifact_uri).parent_artifact_uris, [a.artifact_uri])
                with self.assertRaises(ArtifactCatalogError):
                    store.register(
                        ArtifactRecord(
                            artifact_uri=a.artifact_uri,
                            artifact_type="training",
                            created_at="2026-01-01T00:00:00+00:00",
                            parent_artifact_uris=[b.artifact_uri],
                        )
                    )
                self.assertEqual(len(store.list_by_capability("deployable")), 1)


class IndexerTests(unittest.TestCase):
    def test_index_models_and_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models" / "demo_model"
            models.mkdir(parents=True)
            (models / "config.json").write_text('{"dataset": "ds1"}', encoding="utf-8")
            (models / "feature_importance_studio").mkdir()
            ds = Path(tmp) / "datasets"
            ds.mkdir()
            (ds / "prediction_a.parquet").write_text("x", encoding="utf-8")
            (ds / "training_dataset_fh_abc").mkdir()
            with ArtifactCatalogStore(tmp) as store:
                summary = rebuild_catalog_index(store, tmp)
                self.assertGreaterEqual(summary["models"], 1)
                self.assertGreaterEqual(summary["datasets"], 1)
                self.assertIsNotNone(store.get(model_uri("demo_model")))
                studio = store.get("aruneo://feature_studio/importance/demo_model")
                self.assertIsNotNone(studio)
                self.assertIn(model_uri("demo_model"), studio.parent_artifact_uris)


class TimelineMetricsTests(unittest.TestCase):
    def test_lineage_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with ArtifactCatalogStore(tmp) as store:
                master = "aruneo://master/day/2026-07-31"
                train = training_uri("tb_v1")
                model = model_uri("xgb_v4")
                store.register(
                    ArtifactRecord(
                        artifact_uri=master,
                        artifact_type="master",
                        created_at="2026-07-31T08:00:00+00:00",
                    )
                )
                store.register(
                    ArtifactRecord(
                        artifact_uri=train,
                        artifact_type="training",
                        created_at="2026-07-31T09:00:00+00:00",
                        parent_artifact_uris=[master],
                        metadata={"strategy": "triple_barrier"},
                    )
                )
                store.register(
                    ArtifactRecord(
                        artifact_uri=model,
                        artifact_type="model",
                        created_at="2026-07-31T10:00:00+00:00",
                        parent_artifact_uris=[train],
                        metadata={"feature_set_id": "fs_193"},
                    )
                )
                chain = lineage_chain_uris(store, model)
                self.assertEqual(chain[0], master)
                self.assertEqual(chain[-1], model)
                m = compute_research_metrics(store)
                self.assertEqual(m.best_label_strategy, "triple_barrier")
                self.assertEqual(m.most_reused_feature_set, "fs_193")
                self.assertIsNotNone(m.avg_dataset_to_model_sec)
                self.assertGreaterEqual(m.avg_dataset_to_model_sec, 0)


class ContractExecutorTests(unittest.TestCase):
    def test_suggestion_to_run_registers_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with ArtifactCatalogService(tmp) as svc:
                model = model_uri("base")
                diag = "aruneo://diagnostics/base"
                svc.register(
                    ArtifactRecord(
                        artifact_uri=model,
                        artifact_type="model",
                        created_at="2026-01-01T00:00:00+00:00",
                    )
                )
                svc.register(
                    ArtifactRecord(
                        artifact_uri=diag,
                        artifact_type="diagnostics",
                        created_at="2026-01-02T00:00:00+00:00",
                        parent_artifact_uris=[model],
                    )
                )
                suggestion = {
                    "title": "Ablate Bottom Features",
                    "hypothesis": "Removing weak features may help.",
                    "features": [{"name": "f_a"}, {"name": "f_b"}],
                    "evidence": {"rule": "feature_removal_candidates"},
                    "label_strategy": "triple_barrier",
                    "label_params": {"tp_pct": 0.5},
                }
                contract = svc.create_contract_from_suggestion(
                    suggestion,
                    model_uri_parent=model,
                    diagnostics_uris=[diag],
                )
                self.assertTrue(contract.runnable)
                self.assertTrue(is_runnable(contract))
                self.assertIn(diag, contract.parent_artifact_uris)
                self.assertEqual(contract.actions.get("label_strategy"), "triple_barrier")
                self.assertTrue(contract.evidence_summary)

                result = svc.run_contract(contract, dry_run=True)
                self.assertTrue(result.ok, result.error)
                self.assertEqual(result.status, "completed")
                self.assertGreaterEqual(len(result.produced_uris), 3)
                exp = svc.get(contract.experiment_uri)
                self.assertEqual(exp.status, "completed")
                # Timeline contains experiment + children
                uris = {e.artifact_uri for e in svc.timeline()}
                self.assertIn(contract.experiment_uri, uris)
                metrics = svc.metrics()
                self.assertGreaterEqual(metrics.experiments_run, 1)


class CycleSelfParentTests(unittest.TestCase):
    def test_self_parent_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with ArtifactCatalogStore(tmp) as store:
                uri = experiment_uri("e1")
                with self.assertRaises(ArtifactCatalogError):
                    store.register(
                        ArtifactRecord(
                            artifact_uri=uri,
                            artifact_type="experiment",
                            created_at="2026-01-01T00:00:00+00:00",
                            parent_artifact_uris=[uri],
                        )
                    )


if __name__ == "__main__":
    unittest.main()
