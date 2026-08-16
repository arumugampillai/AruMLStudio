"""Unit tests for Production Validation Phase A (unseen dataset resolve)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock


class ProductionValidationNamingTests(unittest.TestCase):
    def test_identity_hash_stable(self) -> None:
        from chain_replay_ml.production_validation.unseen_dataset import (
            build_unseen_dataset_name,
            unseen_dataset_identity_hash,
        )

        h1 = unseen_dataset_identity_hash(
            master_db_path=r"C:\data\master.db",
            unseen_days=["2026-07-02", "2026-07-01"],
            master_filter={"premium_min": 10, "premium_max": 200},
            parent_dataset="Parent_DS",
        )
        h2 = unseen_dataset_identity_hash(
            master_db_path=r"C:/data/master.db",
            unseen_days=["2026-07-01", "2026-07-02"],
            master_filter={"premium_max": 200, "premium_min": 10},
            parent_dataset="Parent_DS",
        )
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 8)
        name = build_unseen_dataset_name(
            model_name="My Model!",
            identity_hash=h1,
            parent_dataset="Parent_DS",
        )
        self.assertTrue(name.startswith("unseen_Parent_DS_"))
        self.assertTrue(name.endswith(h1))

    def test_name_prefers_parent_slug(self) -> None:
        from chain_replay_ml.production_validation.unseen_dataset import (
            build_unseen_dataset_name,
        )

        name = build_unseen_dataset_name(
            model_name="ModelA",
            identity_hash="abcdef12",
            parent_dataset=None,
        )
        self.assertEqual(name, "unseen_ModelA_abcdef12")


class ProductionValidationResolveTests(unittest.TestCase):
    def _write_pkg(
        self,
        data_dir: str,
        model_name: str = "PV_Model",
        *,
        seen_days: list[str],
        master_rel: str,
        parent: str = "ParentTrain",
    ) -> str:
        from chain_replay_ml.training.paths import model_package_dir, safe_model_name

        safe = safe_model_name(model_name)
        pkg = model_package_dir(data_dir, safe)
        os.makedirs(pkg, exist_ok=True)
        snap = {
            "dataset_name": parent,
            "master_db_path": master_rel,
            "trading_day_labels": ",".join(seen_days),
            "trading_days": len(seen_days),
            "exported_dates": list(seen_days),
            "master_filter": {"premium_enabled": False},
        }
        with open(os.path.join(pkg, "dataset_build_snapshot.json"), "w", encoding="utf-8") as fh:
            json.dump(snap, fh)
        with open(os.path.join(pkg, "training_config.json"), "w", encoding="utf-8") as fh:
            json.dump({"dataset": parent, "dataset_name": parent}, fh)
        with open(os.path.join(pkg, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump({"model_name": safe}, fh)
        return safe

    def _fake_master_days(self, path: str, days: list[str]) -> dict[str, int]:
        del path
        return {d: 100 for d in days}

    def test_resolve_reuses_valid_registry_dataset(self) -> None:
        from chain_replay_ml.dataset_builder.writer import datasets_dir
        from chain_replay_ml.production_validation import resolve_unseen_dataset_for_model

        pipe_feats = [f"pipe_feat_{i}" for i in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            master_path = os.path.join(tmp, "nifty_master_3s.db")
            with open(master_path, "wb") as fh:
                fh.write(b"sqlite")
            seen = ["2026-07-01", "2026-07-02"]
            master_days = ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04"]
            unseen = ["2026-07-03", "2026-07-04"]
            model = self._write_pkg(
                tmp,
                seen_days=seen,
                master_rel=os.path.basename(master_path),
            )

            with mock.patch(
                "chain_replay_ml.model_lab.prediction_dataset_type.load_master_day_row_counts",
                side_effect=lambda p: self._fake_master_days(p, master_days),
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_dataset_type.resolve_master_db_path_for_lab",
                return_value=os.path.abspath(master_path),
            ), mock.patch(
                "chain_replay_ml.dataset_builder.feature_sources_catalog.pipeline_feature_names",
                return_value=pipe_feats,
            ):
                pending = resolve_unseen_dataset_for_model(
                    data_dir=tmp,
                    model_name=model,
                    create_if_missing=False,
                )
                self.assertEqual(pending.status, "pending")
                ds_name = str(pending.dataset_name)
                identity = str(pending.identity_hash)

                out_dir = datasets_dir(tmp)
                os.makedirs(out_dir, exist_ok=True)
                pq = os.path.join(out_dir, f"{ds_name}.parquet")
                js = os.path.join(out_dir, f"{ds_name}.json")
                with open(pq, "wb") as fh:
                    fh.write(b"PAR1")
                with open(js, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "dataset_name": ds_name,
                            "dataset_kind": "unseen",
                            "keep_pipeline_owned": True,
                            "feature_columns": ["reg_a", "reg_b"] + pipe_feats,
                            "days": [{"trading_day": d} for d in unseen],
                            "production_validation": {
                                "role": "unseen",
                                "identity_hash": identity,
                                "unseen_days": unseen,
                                "include_pipeline": True,
                            },
                        },
                        fh,
                    )

                with mock.patch(
                    "chain_replay_ml.dataset_builder.analysis_dataset_export.create_analysis_dataset"
                ) as create_export:
                    result = resolve_unseen_dataset_for_model(
                        data_dir=tmp,
                        model_name=model,
                        create_if_missing=True,
                    )
                    create_export.assert_not_called()

            self.assertTrue(result.ok)
            self.assertTrue(result.reused)
            self.assertFalse(result.created)
            self.assertEqual(result.dataset_name, ds_name)
            self.assertEqual(result.unseen_days, unseen)
            self.assertEqual(result.status, "ready")
            self.assertIn("compute coming", result.compute_note)

    def test_resolve_rejects_registry_only_unseen(self) -> None:
        """Incomplete ~registry-only unseen_* must not be reused."""
        from chain_replay_ml.dataset_builder.writer import datasets_dir
        from chain_replay_ml.production_validation import resolve_unseen_dataset_for_model

        pipe_feats = [f"pipe_feat_{i}" for i in range(20)]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            master_path = os.path.join(tmp, "nifty_master_3s.db")
            with open(master_path, "wb") as fh:
                fh.write(b"sqlite")
            seen = ["2026-07-01"]
            master_days = ["2026-07-01", "2026-07-03"]
            unseen = ["2026-07-03"]
            model = self._write_pkg(
                tmp,
                seen_days=seen,
                master_rel=os.path.basename(master_path),
            )

            fake_store = mock.MagicMock()
            fake_store.get_meta.return_value = {
                "market": "NIFTY",
                "sampling_interval_sec": 3,
            }
            fake_store_cls = mock.MagicMock(return_value=fake_store)

            with mock.patch(
                "chain_replay_ml.model_lab.prediction_dataset_type.load_master_day_row_counts",
                side_effect=lambda p: self._fake_master_days(p, master_days),
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_dataset_type.resolve_master_db_path_for_lab",
                return_value=os.path.abspath(master_path),
            ), mock.patch(
                "chain_replay_ml.dataset_builder.feature_sources_catalog.pipeline_feature_names",
                return_value=pipe_feats,
            ), mock.patch(
                "chain_replay_ml.dataset_builder.master_store.MasterStore",
                fake_store_cls,
            ):
                pending = resolve_unseen_dataset_for_model(
                    data_dir=tmp,
                    model_name=model,
                    create_if_missing=False,
                )
                ds_name = str(pending.dataset_name)
                identity = str(pending.identity_hash)
                out_dir = datasets_dir(tmp)
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, f"{ds_name}.parquet"), "wb") as fh:
                    fh.write(b"PAR1")
                with open(os.path.join(out_dir, f"{ds_name}.json"), "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "dataset_name": ds_name,
                            "dataset_kind": "unseen",
                            "feature_columns": [f"reg_{i}" for i in range(206)],
                            "feature_count": 206,
                            "days": [{"trading_day": d} for d in unseen],
                            "production_validation": {
                                "role": "unseen",
                                "identity_hash": identity,
                                "unseen_days": unseen,
                            },
                        },
                        fh,
                    )

                created_payload = {
                    "dataset_name": ds_name,
                    "json_path": os.path.join(out_dir, f"{ds_name}.json"),
                    "parquet_path": os.path.join(out_dir, f"{ds_name}.parquet"),
                    "feature_count": 400,
                    "pipeline_present": 20,
                }

                def _fake_create(*_a, **kwargs):
                    self.assertTrue(kwargs.get("include_pipeline"))
                    self.assertEqual(kwargs.get("dataset_kind"), "unseen")
                    self.assertTrue(str(kwargs.get("dataset_name") or "").startswith("unseen_"))
                    # Recreate writes a pipeline-complete meta for stamp/follow-up.
                    with open(created_payload["parquet_path"], "wb") as fh:
                        fh.write(b"PAR1")
                    with open(created_payload["json_path"], "w", encoding="utf-8") as fh:
                        json.dump(
                            {
                                "dataset_name": ds_name,
                                "dataset_kind": "unseen",
                                "keep_pipeline_owned": True,
                                "feature_columns": [f"reg_{i}" for i in range(206)] + pipe_feats,
                                "feature_count": 226,
                            },
                            fh,
                        )
                    return created_payload

                with mock.patch(
                    "chain_replay_ml.dataset_builder.analysis_dataset_export.create_analysis_dataset",
                    side_effect=_fake_create,
                ) as create_export:
                    result = resolve_unseen_dataset_for_model(
                        data_dir=tmp,
                        model_name=model,
                        create_if_missing=True,
                    )
                    create_export.assert_called_once()

            self.assertTrue(result.ok)
            self.assertTrue(result.created)
            self.assertFalse(result.reused)
            self.assertEqual(result.status, "ready")
            self.assertIn("pipeline", result.message.lower())

    def test_resolve_empty_when_no_unseen(self) -> None:
        from chain_replay_ml.production_validation import resolve_unseen_dataset_for_model

        with tempfile.TemporaryDirectory() as tmp:
            master_path = os.path.join(tmp, "master.db")
            with open(master_path, "wb") as fh:
                fh.write(b"x")
            days = ["2026-07-01", "2026-07-02"]
            model = self._write_pkg(
                tmp,
                seen_days=days,
                master_rel=os.path.basename(master_path),
            )
            with mock.patch(
                "chain_replay_ml.model_lab.prediction_dataset_type.load_master_day_row_counts",
                return_value={d: 10 for d in days},
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_dataset_type.resolve_master_db_path_for_lab",
                return_value=os.path.abspath(master_path),
            ):
                result = resolve_unseen_dataset_for_model(
                    data_dir=tmp,
                    model_name=model,
                    create_if_missing=True,
                )
            self.assertTrue(result.ok)
            self.assertEqual(result.status, "empty")
            self.assertEqual(result.unseen_days, [])

    def test_create_if_missing_false_pending(self) -> None:
        from chain_replay_ml.production_validation import resolve_unseen_dataset_for_model

        with tempfile.TemporaryDirectory() as tmp:
            master_path = os.path.join(tmp, "master.db")
            with open(master_path, "wb") as fh:
                fh.write(b"x")
            seen = ["2026-07-01"]
            master_days = ["2026-07-01", "2026-07-10"]
            model = self._write_pkg(
                tmp,
                seen_days=seen,
                master_rel="master.db",
            )
            with mock.patch(
                "chain_replay_ml.model_lab.prediction_dataset_type.load_master_day_row_counts",
                return_value={d: 10 for d in master_days},
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_dataset_type.resolve_master_db_path_for_lab",
                return_value=os.path.abspath(master_path),
            ):
                result = resolve_unseen_dataset_for_model(
                    data_dir=tmp,
                    model_name=model,
                    create_if_missing=False,
                )
            self.assertTrue(result.ok)
            self.assertEqual(result.status, "pending")
            self.assertTrue(str(result.dataset_name or "").startswith("unseen_"))
            self.assertEqual(result.unseen_days, ["2026-07-10"])

    def test_realistic_pl0005_lineage_resolve(self) -> None:
        """Test realistic lineage propagation for PL_0005 parent dataset."""
        from chain_replay_ml.dataset_builder.writer import datasets_dir
        from chain_replay_ml.production_validation import resolve_unseen_dataset_for_model

        with tempfile.TemporaryDirectory() as tmp:
            master_path = os.path.join(tmp, "nifty_master_3s.db")
            with open(master_path, "wb") as fh:
                fh.write(b"")
            seen = ["2026-07-01", "2026-07-31"]
            master_days = ["2026-07-01", "2026-07-31", "2026-08-01"]
            parent_name = "analysis_PL0005_198r_447p_6s_20260814_221827"
            model_name = "Future_LTP_5m_WF_1168f_XGB_2243_14"

            # 1. Create parent dataset metadata
            ds_dir = datasets_dir(tmp)
            os.makedirs(ds_dir, exist_ok=True)
            parent_json = os.path.join(ds_dir, f"{parent_name}.json")
            with open(parent_json, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "dataset_name": parent_name,
                        "pipeline_id": "PL_0005",
                        "pipeline_name": "Pipeline_005",
                        "pipeline_type": "auto",
                        "pipeline_snapshot_id": "ca5945f58f87e96e",
                        "feature_project_id": "all",
                        "include_pipeline": True,
                        "include_registry": True,
                        "registry_export_features": ["spot", "ltp"],
                        "base_pipeline_export_features": ["fwd_ret_5m"],
                        "pipeline_provenance": {
                            "pipeline_id": "PL_0005",
                            "candidate_features": ["custom_exp_1", "custom_exp_2"],
                        },
                        "days": [{"trading_day": "2026-07-01"}, {"trading_day": "2026-07-31"}],
                    },
                    fh,
                )

            # 2. Create model package
            safe = self._write_pkg(
                tmp,
                model_name=model_name,
                seen_days=seen,
                master_rel=os.path.basename(master_path),
                parent=parent_name,
            )

            created_kwargs: dict = {}

            def fake_create_analysis(data_dir: str, **kwargs):
                created_kwargs.update(kwargs)
                out_pq = os.path.join(ds_dir, f"{kwargs['dataset_name']}.parquet")
                out_js = os.path.join(ds_dir, f"{kwargs['dataset_name']}.json")
                with open(out_pq, "wb") as f:
                    f.write(b"pq")
                with open(out_js, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "dataset_name": kwargs["dataset_name"],
                            "dataset_kind": "unseen",
                            "feature_project_id": kwargs.get("feature_project_id"),
                            "pipeline_id": kwargs.get("pipeline_id"),
                            "include_pipeline": kwargs.get("include_pipeline"),
                            "include_registry": kwargs.get("include_registry"),
                            "days": [{"trading_day": "2026-08-01"}],
                        },
                        f,
                    )
                return {
                    "dataset_name": kwargs["dataset_name"],
                    "json_path": out_js,
                    "parquet_path": out_pq,
                    "feature_count": 1186,
                    "pipeline_present": 447,
                }

            fake_store = mock.MagicMock()
            fake_store.get_meta.return_value = {"market": "NIFTY", "sampling_interval_sec": 3}

            with mock.patch(
                "chain_replay_ml.model_lab.prediction_dataset_type.load_master_day_row_counts",
                return_value={d: 100 for d in master_days},
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_dataset_type.resolve_master_db_path_for_lab",
                return_value=os.path.abspath(master_path),
            ), mock.patch(
                "chain_replay_ml.dataset_builder.master_store.MasterStore",
                return_value=fake_store,
            ), mock.patch(
                "chain_replay_ml.dataset_builder.analysis_dataset_export.create_analysis_dataset",
                side_effect=fake_create_analysis,
            ):
                res = resolve_unseen_dataset_for_model(
                    data_dir=tmp,
                    model_name=safe,
                    create_if_missing=True,
                )

            self.assertTrue(res.ok, msg=res.error)
            self.assertEqual(res.status, "ready")
            self.assertTrue(res.created)
            self.assertFalse(res.reused)

            # Verify arguments forwarded to create_analysis_dataset
            self.assertEqual(created_kwargs.get("pipeline_id"), "PL_0005")
            self.assertEqual(created_kwargs.get("feature_project_id"), "all")
            self.assertTrue(created_kwargs.get("include_pipeline"))
            self.assertTrue(created_kwargs.get("include_registry"))

    def test_reuse_and_stale_rejection_rules(self) -> None:
        """Test reuse on matching lineage, and rejection on snapshot, pipeline_id, or project divergence."""
        from chain_replay_ml.production_validation.unseen_dataset import (
            _existing_unseen_valid,
            _has_pipeline_features,
            unseen_dataset_identity_hash,
        )

        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = os.path.join(tmp, "datasets")
            os.makedirs(ds_dir, exist_ok=True)
            ds_name = "unseen_test_ds"
            js_path = os.path.join(ds_dir, f"{ds_name}.json")
            pq_path = os.path.join(ds_dir, f"{ds_name}.parquet")
            with open(pq_path, "wb") as f:
                f.write(b"pq")

            ident = unseen_dataset_identity_hash(
                master_db_path=os.path.join(tmp, "master.db"),
                unseen_days=["2026-08-01"],
                parent_dataset="Parent_PL0005",
                feature_project_id="all",
                pipeline_id="PL_0005",
                pipeline_snapshot_id="ca5945f58f87e96e",
            )

            base_meta = {
                "dataset_name": ds_name,
                "dataset_kind": "unseen",
                "days": [{"trading_day": "2026-08-01"}],
                "feature_project_id": "all",
                "pipeline_id": "PL_0005",
                "pipeline_snapshot_id": "ca5945f58f87e96e",
                "production_validation": {
                    "role": "unseen",
                    "identity_hash": ident,
                    "feature_project_id": "all",
                    "pipeline_id": "PL_0005",
                    "pipeline_snapshot_id": "ca5945f58f87e96e",
                },
            }

            with open(js_path, "w", encoding="utf-8") as f:
                json.dump(base_meta, f)

            with mock.patch(
                "chain_replay_ml.production_validation.unseen_dataset._has_pipeline_features",
                return_value=True,
            ):
                # A. Same pipeline_id + same snapshot -> reuse allowed
                ok = _existing_unseen_valid(
                    data_dir=tmp,
                    dataset_name=ds_name,
                    expected_days=["2026-08-01"],
                    identity_hash=ident,
                    expected_feature_project_id="all",
                    expected_pipeline_id="PL_0005",
                    expected_pipeline_snapshot_id="ca5945f58f87e96e",
                )
                self.assertIsNotNone(ok)

                # B. Same pipeline_id + different snapshot -> reject reuse
                rej_snap = _existing_unseen_valid(
                    data_dir=tmp,
                    dataset_name=ds_name,
                    expected_days=["2026-08-01"],
                    identity_hash=ident,
                    expected_feature_project_id="all",
                    expected_pipeline_id="PL_0005",
                    expected_pipeline_snapshot_id="diff_snapshot_9999",
                )
                self.assertIsNone(rej_snap)

                # C. Different pipeline_id -> reject reuse
                rej_pid = _existing_unseen_valid(
                    data_dir=tmp,
                    dataset_name=ds_name,
                    expected_days=["2026-08-01"],
                    identity_hash=ident,
                    expected_feature_project_id="all",
                    expected_pipeline_id="PL_0002",
                    expected_pipeline_snapshot_id="ca5945f58f87e96e",
                )
                self.assertIsNone(rej_pid)

                # D. Different feature_project_id -> reject reuse
                rej_fpid = _existing_unseen_valid(
                    data_dir=tmp,
                    dataset_name=ds_name,
                    expected_days=["2026-08-01"],
                    identity_hash=ident,
                    expected_feature_project_id="nifty_classification",
                    expected_pipeline_id="PL_0005",
                    expected_pipeline_snapshot_id="ca5945f58f87e96e",
                )
                self.assertIsNone(rej_fpid)

    def test_pipeline_enabled_but_pipeline_id_missing_error(self) -> None:
        """E. When pipeline is enabled on parent dataset but pipeline_id is missing, returns clear error."""
        from chain_replay_ml.dataset_builder.writer import datasets_dir
        from chain_replay_ml.production_validation import resolve_unseen_dataset_for_model

        with tempfile.TemporaryDirectory() as tmp:
            master_path = os.path.join(tmp, "nifty_master_3s.db")
            with open(master_path, "wb") as fh:
                fh.write(b"")
            seen = ["2026-07-01"]
            master_days = ["2026-07-01", "2026-07-02"]
            parent_name = "parent_missing_pipe_id"
            model_name = "Model_Pipe_Missing"

            ds_dir = datasets_dir(tmp)
            os.makedirs(ds_dir, exist_ok=True)
            with open(os.path.join(ds_dir, f"{parent_name}.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "dataset_name": parent_name,
                        "feature_project_id": "all",
                        "include_pipeline": True,
                        "pipeline_id": "",
                        "days": [{"trading_day": "2026-07-01"}],
                    },
                    fh,
                )

            safe = self._write_pkg(
                tmp,
                model_name=model_name,
                seen_days=seen,
                master_rel=os.path.basename(master_path),
                parent=parent_name,
            )

            fake_store = mock.MagicMock()
            fake_store.get_meta.return_value = {"market": "NIFTY", "sampling_interval_sec": 3}

            with mock.patch(
                "chain_replay_ml.model_lab.prediction_dataset_type.load_master_day_row_counts",
                return_value={d: 100 for d in master_days},
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_dataset_type.resolve_master_db_path_for_lab",
                return_value=os.path.abspath(master_path),
            ), mock.patch(
                "chain_replay_ml.dataset_builder.master_store.MasterStore",
                return_value=fake_store,
            ):
                res = resolve_unseen_dataset_for_model(
                    data_dir=tmp,
                    model_name=safe,
                    create_if_missing=True,
                )

            self.assertFalse(res.ok)
            self.assertEqual(res.status, "error")
            self.assertIn("Pipeline Features is enabled", res.error)


class ProductionValidationRulesTests(unittest.TestCase):
    def test_rank_by_abs_importance(self) -> None:
        from chain_replay_ml.production_validation.rules import rank_by_abs_importance

        ranks = rank_by_abs_importance({"a": 0.1, "b": -0.9, "c": 0.5})
        self.assertEqual(ranks["b"], 1)  # abs 0.9
        self.assertEqual(ranks["c"], 2)
        self.assertEqual(ranks["a"], 3)

    def test_rank_change_and_importance_difference(self) -> None:
        from chain_replay_ml.production_validation.rules import (
            build_feature_rows,
            importance_difference,
        )

        self.assertAlmostEqual(importance_difference(1.0, 0.4), -0.6)
        rows, meta = build_feature_rows(
            holdout_importance={"hot": 1.0, "mid": 0.5, "cold": 0.1},
            unseen_importance={"hot": 0.05, "mid": 0.5, "cold": 0.9},
        )
        by_feat = {r["feature"]: r for r in rows}
        self.assertEqual(by_feat["hot"]["holdout_rank"], 1)
        self.assertEqual(by_feat["hot"]["unseen_rank"], 3)
        self.assertEqual(by_feat["hot"]["rank_change"], 1 - 3)  # −2 = less important
        self.assertAlmostEqual(by_feat["hot"]["importance_difference"], -0.95)
        self.assertEqual(by_feat["cold"]["rank_change"], 3 - 1)  # rose on unseen
        self.assertTrue(meta.get("degraded_to_rank_imp_only"))

    def test_recommendation_remove_without_drift(self) -> None:
        from chain_replay_ml.production_validation.rules import recommend_feature

        # Large rank drop (−6) + large relative imp drop (0.8) → REMOVE
        rec, detail = recommend_feature(
            rank_change=-6,
            holdout_importance=1.0,
            importance_diff=-0.8,
        )
        self.assertEqual(rec, "REMOVE")
        self.assertFalse(detail["drift_signals_available"])

        # Medium rank drop only → WATCH
        rec2, _ = recommend_feature(
            rank_change=-3,
            holdout_importance=1.0,
            importance_diff=-0.05,
        )
        self.assertEqual(rec2, "WATCH")

        # Stable → KEEP
        rec3, _ = recommend_feature(
            rank_change=0,
            holdout_importance=1.0,
            importance_diff=-0.05,
        )
        self.assertEqual(rec3, "KEEP")

    def test_recommendation_requires_drift_when_available(self) -> None:
        from chain_replay_ml.production_validation.rules import recommend_feature

        # Large rank+imp but low drift → not REMOVE (needs high drift when available)
        rec, detail = recommend_feature(
            rank_change=-6,
            holdout_importance=1.0,
            importance_diff=-0.8,
            feature_drift=0.05,
            ks_statistic=0.05,
            wasserstein_normalized=0.1,
        )
        self.assertTrue(detail["drift_signals_available"])
        self.assertEqual(rec, "WATCH")  # rank+imp severe but drift low → WATCH path

        # Same + high KS → REMOVE
        rec2, _ = recommend_feature(
            rank_change=-6,
            holdout_importance=1.0,
            importance_diff=-0.8,
            ks_statistic=0.40,
        )
        self.assertEqual(rec2, "REMOVE")

    def test_feature_validation_summary(self) -> None:
        from chain_replay_ml.production_validation.rules import (
            build_dual_confidence,
            build_feature_rows,
            build_feature_validation_summary,
        )

        # 8 features so a top→bottom drop exceeds REMOVE rank threshold (−5).
        holdout = {f"f{i}": float(8 - i) for i in range(8)}
        unseen = dict(holdout)
        unseen["f0"] = 0.01  # was most important → large rank + magnitude drop
        rows, _ = build_feature_rows(
            holdout_importance=holdout,
            unseen_importance=unseen,
            drift_by_feature={
                "f0": {
                    "drift": 0.6,
                    "ks_statistic": 0.4,
                    "wasserstein_normalized": 1.2,
                }
            },
        )
        summary = build_feature_validation_summary(rows)
        self.assertEqual(summary["feature_count"], 8)
        self.assertEqual(
            summary["keep_count"] + summary["watch_count"] + summary["remove_count"],
            8,
        )
        self.assertIn("average_rank_change", summary)
        self.assertIn("median_rank_change", summary)
        self.assertIn("stable_features_pct", summary)
        self.assertGreaterEqual(summary["stable_features_pct"], 0.0)
        self.assertLessEqual(summary["stable_features_pct"], 100.0)

        by_feat = {r["feature"]: r for r in rows}
        self.assertEqual(by_feat["f0"]["holdout_rank"], 1)
        self.assertEqual(by_feat["f0"]["unseen_rank"], 8)
        self.assertEqual(by_feat["f0"]["rank_change"], -7)
        self.assertEqual(by_feat["f0"]["recommendation"], "REMOVE")

        dual = build_dual_confidence(
            rows, unseen_day_count=1, feature_summary=summary
        )
        self.assertIn(dual["diagnosis"]["label"], ("Overfit", "Fragile", "Stable"))
        self.assertEqual(dual["production_confirmation"]["unseen_days_tested"], 1)
        self.assertIn("median_rank_change", dual["diagnosis"])
        self.assertNotIn("median_collapse_pct", dual["diagnosis"])
        self.assertIn("Tested 1 unseen day", dual["production_confirmation"]["explanation"])
        self.assertIn("remove_rank_drop", dual["thresholds"])

    def test_enrich_legacy_collapse_rows_without_recompute(self) -> None:
        """Pre-v1.1 artifacts (importances + collapse_pct) get ranks on load."""
        import tempfile

        from chain_replay_ml.production_validation.rules import (
            enrich_comparison_rows_from_importances,
        )
        from chain_replay_ml.production_validation.writer import (
            load_validation_artifacts,
            write_validation_artifacts,
        )

        legacy_rows = [
            {
                "feature": "hot",
                "holdout_importance": 1.0,
                "unseen_importance": 0.05,
                "collapse_pct": 95.0,
                "recommendation": "REMOVE",
            },
            {
                "feature": "mid",
                "holdout_importance": 0.5,
                "unseen_importance": 0.5,
                "collapse_pct": 0.0,
                "recommendation": "KEEP",
            },
            {
                "feature": "cold",
                "holdout_importance": 0.1,
                "unseen_importance": 0.9,
                "collapse_pct": -800.0,
                "recommendation": "REMOVE",
            },
        ]
        enriched, did = enrich_comparison_rows_from_importances(legacy_rows)
        self.assertTrue(did)
        by_feat = {r["feature"]: r for r in enriched}
        self.assertEqual(by_feat["hot"]["holdout_rank"], 1)
        self.assertEqual(by_feat["hot"]["unseen_rank"], 3)
        self.assertEqual(by_feat["hot"]["rank_change"], -2)
        self.assertAlmostEqual(by_feat["hot"]["importance_difference"], -0.95)
        self.assertNotIn("collapse_pct", by_feat["hot"])
        # Already-ranked rows are left alone.
        again, did2 = enrich_comparison_rows_from_importances(enriched)
        self.assertFalse(did2)
        self.assertEqual(again[0]["holdout_rank"], enriched[0]["holdout_rank"])

        with tempfile.TemporaryDirectory() as tmp:
            write_validation_artifacts(
                tmp,
                comparison=legacy_rows,
                summary={
                    "model_name": "M",
                    "unseen_day_count": 1,
                    "feature_validation": {
                        "keep_count": 1,
                        "watch_count": 0,
                        "remove_count": 2,
                        "average_collapse_pct": 50.0,
                        "median_collapse_pct": 95.0,
                    },
                    "diagnosis": {
                        "label": "Overfit",
                        "confidence_pct": 80,
                        "median_collapse_pct": 95.0,
                    },
                },
                run_meta={"studio_version": "1.0.0-collapse"},
            )
            loaded = load_validation_artifacts(tmp)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertTrue(loaded.get("rank_fields_enriched"))
            row0 = loaded["rows"][0]
            self.assertIn("holdout_rank", row0)
            self.assertIn("importance_difference", row0)
            self.assertNotIn("collapse_pct", row0)
            fv = loaded["summary"]["feature_validation"]
            self.assertIn("average_rank_change", fv)
            self.assertIn("median_rank_change", fv)
            self.assertNotIn("average_collapse_pct", fv)
            self.assertIn("median_rank_change", loaded["summary"]["diagnosis"])


class ProductionValidationComputeTests(unittest.TestCase):
    def test_compute_holdout_vs_unseen_permutation(self) -> None:
        import numpy as np
        import pandas as pd
        from xgboost import XGBRegressor

        from chain_replay_ml.dataset_builder.writer import datasets_dir
        from chain_replay_ml.production_validation import run_production_validation
        from chain_replay_ml.training.paths import model_package_dir, safe_model_name

        features = ["f_signal", "f_noise"]
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(11)
            n = 80
            signal = rng.normal(size=n)
            noise = rng.normal(size=n)
            y = signal * 2.5 + rng.normal(scale=0.05, size=n)
            X = pd.DataFrame({"f_signal": signal, "f_noise": noise})

            model_name = "PV_PhaseB_Tiny"
            safe = safe_model_name(model_name)
            pkg = model_package_dir(tmp, safe)
            os.makedirs(pkg, exist_ok=True)
            booster = XGBRegressor(
                n_estimators=20,
                max_depth=2,
                learning_rate=0.3,
                objective="reg:squarederror",
                verbosity=0,
            )
            booster.fit(X, y)
            booster.save_model(os.path.join(pkg, "model.ubj"))
            with open(os.path.join(pkg, "config.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "algorithm": "xgboost",
                        "target": "future_ltp_5m",
                        "prediction_type": "regression",
                        "selected_features": features,
                        "features": features,
                        "dataset": "train_parent",
                    },
                    fh,
                )
            with open(os.path.join(pkg, "metadata.json"), "w", encoding="utf-8") as fh:
                json.dump({"model_name": safe}, fh)

            # Unseen registry dataset (whole day rows).
            unseen_name = "unseen_pv_tiny_abcdef12"
            out_dir = datasets_dir(tmp)
            os.makedirs(out_dir, exist_ok=True)
            n_u = 60
            sig_u = rng.normal(size=n_u)
            noise_u = rng.normal(size=n_u)
            # Collapse signal importance: weaker relationship on unseen.
            y_u = sig_u * 0.4 + rng.normal(scale=0.4, size=n_u)
            df_u = pd.DataFrame(
                {
                    "trading_day": ["2026-07-30"] * n_u,
                    "f_signal": sig_u,
                    "f_noise": noise_u,
                    "future_ltp_5m": y_u,
                }
            )
            pq = os.path.join(out_dir, f"{unseen_name}.parquet")
            js = os.path.join(out_dir, f"{unseen_name}.json")
            df_u.to_parquet(pq, index=False)
            with open(js, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "dataset_name": unseen_name,
                        "dataset_kind": "unseen",
                        "prediction_target_columns": ["future_ltp_5m"],
                        "feature_columns": features,
                        "days": [{"trading_day": "2026-07-30"}],
                    },
                    fh,
                )

            pv_dir = os.path.join(pkg, "production_validation")
            os.makedirs(pv_dir, exist_ok=True)
            with open(os.path.join(pv_dir, "unseen_dataset.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "ok": True,
                        "model_name": safe,
                        "dataset_name": unseen_name,
                        "parquet_path": pq,
                        "json_path": js,
                        "status": "ready",
                        "unseen_days": ["2026-07-30"],
                        "unseen_day_count": 1,
                        "seen_day_count": 2,
                        "compute_note": "compute coming",
                    },
                    fh,
                )

            def _fake_holdout(**_kwargs):
                return X.copy(), pd.Series(y), features, {
                    "dataset": "train_parent",
                    "target": "future_ltp_5m",
                    "holdout_start": 0,
                    "holdout_stop": n,
                    "holdout_rows": n,
                    "feature_count": len(features),
                    "dataset_load": {},
                    "prediction_type": "regression",
                }

            with mock.patch(
                "chain_replay_ml.production_validation.compute._load_holdout_xy",
                side_effect=_fake_holdout,
            ), mock.patch(
                "chain_replay_ml.production_validation.compute.load_model_detail",
                return_value={"model_name": safe},
            ):
                result = run_production_validation(
                    data_dir=tmp,
                    model_name=safe,
                    holdout_max_rows=None,
                    unseen_max_rows=None,
                    permutation_n_repeats=3,
                    resolve_unseen_if_needed=False,
                )

            self.assertTrue(result.ok, msg=result.error)
            self.assertEqual(len(result.rows), 2)
            by_feat = {r["feature"]: r for r in result.rows}
            self.assertIn("f_signal", by_feat)
            self.assertIn(by_feat["f_signal"]["recommendation"], ("KEEP", "WATCH", "REMOVE"))
            self.assertIn("holdout_rank", by_feat["f_signal"])
            self.assertIn("unseen_rank", by_feat["f_signal"])
            self.assertIn("rank_change", by_feat["f_signal"])
            self.assertIn("importance_difference", by_feat["f_signal"])
            self.assertNotIn("collapse_pct", by_feat["f_signal"])
            self.assertIn("feature_validation", result.summary)
            self.assertEqual(
                result.summary["feature_validation"]["feature_count"],
                2,
            )
            self.assertTrue(os.path.isfile(os.path.join(pv_dir, "comparison.json")))
            self.assertTrue(os.path.isfile(os.path.join(pv_dir, "summary.json")))
            self.assertEqual(
                result.summary.get("production_confirmation", {}).get("unseen_days_tested"),
                1,
            )
            self.assertEqual(result.meta.get("unseen_coverage"), "whole_unseen_days")
            self.assertIn(result.meta.get("inference_device"), ("CUDA", "CPU"))
            self.assertIn("gpu_active", result.meta)
            self.assertIn(result.summary.get("inference_device"), ("CUDA", "CPU"))
            self.assertIn("rank_change_formula", result.meta)
            self.assertNotIn("collapse_formula", result.meta)
    def test_compute_emits_progress_stages(self) -> None:
        import numpy as np
        import pandas as pd
        from xgboost import XGBRegressor

        from chain_replay_ml.dataset_builder.writer import datasets_dir
        from chain_replay_ml.production_validation import run_production_validation
        from chain_replay_ml.training.paths import model_package_dir, safe_model_name

        features = ["f_a", "f_b"]
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(3)
            n = 40
            X = pd.DataFrame(
                {"f_a": rng.normal(size=n), "f_b": rng.normal(size=n)}
            )
            y = X["f_a"] * 1.5 + rng.normal(scale=0.1, size=n)
            safe = safe_model_name("PV_Progress")
            pkg = model_package_dir(tmp, safe)
            os.makedirs(pkg, exist_ok=True)
            booster = XGBRegressor(
                n_estimators=8, max_depth=2, learning_rate=0.3, verbosity=0
            )
            booster.fit(X, y)
            booster.save_model(os.path.join(pkg, "model.ubj"))
            with open(os.path.join(pkg, "config.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "algorithm": "xgboost",
                        "target": "future_ltp_5m",
                        "prediction_type": "regression",
                        "selected_features": features,
                        "features": features,
                        "dataset": "train_parent",
                    },
                    fh,
                )
            unseen_name = "unseen_pv_prog_abcdef12"
            out_dir = datasets_dir(tmp)
            os.makedirs(out_dir, exist_ok=True)
            n_u = 30
            df_u = pd.DataFrame(
                {
                    "trading_day": ["2026-07-30"] * n_u,
                    "f_a": rng.normal(size=n_u),
                    "f_b": rng.normal(size=n_u),
                    "future_ltp_5m": rng.normal(size=n_u),
                }
            )
            pq = os.path.join(out_dir, f"{unseen_name}.parquet")
            js = os.path.join(out_dir, f"{unseen_name}.json")
            df_u.to_parquet(pq, index=False)
            with open(js, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "dataset_name": unseen_name,
                        "dataset_kind": "unseen",
                        "prediction_target_columns": ["future_ltp_5m"],
                        "feature_columns": features,
                        "days": [{"trading_day": "2026-07-30"}],
                    },
                    fh,
                )
            pv_dir = os.path.join(pkg, "production_validation")
            os.makedirs(pv_dir, exist_ok=True)
            with open(os.path.join(pv_dir, "unseen_dataset.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "ok": True,
                        "model_name": safe,
                        "dataset_name": unseen_name,
                        "parquet_path": pq,
                        "json_path": js,
                        "status": "ready",
                        "unseen_days": ["2026-07-30"],
                        "unseen_day_count": 1,
                        "compute_note": "compute coming",
                    },
                    fh,
                )

            events: list[dict] = []

            def _progress(info: dict) -> None:
                events.append(dict(info))

            with mock.patch(
                "chain_replay_ml.production_validation.compute._load_holdout_xy",
                return_value=(
                    X.copy(),
                    pd.Series(y),
                    features,
                    {
                        "dataset": "train_parent",
                        "target": "future_ltp_5m",
                        "holdout_rows": n,
                        "prediction_type": "regression",
                    },
                ),
            ), mock.patch(
                "chain_replay_ml.production_validation.compute.load_model_detail",
                return_value={"model_name": safe},
            ):
                result = run_production_validation(
                    data_dir=tmp,
                    model_name=safe,
                    permutation_n_repeats=2,
                    resolve_unseen_if_needed=False,
                    progress=_progress,
                )

            self.assertTrue(result.ok, msg=result.error)
            stages = [e.get("stage") for e in events]
            self.assertIn("load_model", stages)
            self.assertIn("configure_inference", stages)
            self.assertIn("load_holdout", stages)
            self.assertIn("load_unseen", stages)
            self.assertIn("permutation_holdout", stages)
            self.assertIn("permutation_unseen", stages)
            self.assertIn("done", stages)
            # Feature-level ticks keep holdout/unseen labels (not bare "permutation").
            perm_events = [
                e for e in events if e.get("stage") == "permutation_holdout" and e.get("feature")
            ]
            self.assertGreaterEqual(len(perm_events), 1)
            cfg = next(e for e in events if e.get("stage") == "configure_inference" and e.get("device"))
            self.assertIn(cfg.get("device"), ("CUDA", "CPU"))

    def test_missing_selected_feature_fails_clearly(self) -> None:
        import numpy as np
        import pandas as pd

        from chain_replay_ml.dataset_builder.writer import datasets_dir
        from chain_replay_ml.production_validation.load_unseen import load_unseen_xy

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = datasets_dir(tmp)
            os.makedirs(out_dir, exist_ok=True)
            name = "unseen_miss_feat"
            df = pd.DataFrame(
                {
                    "trading_day": ["2026-07-30"] * 20,
                    "f_ok": np.arange(20, dtype=float),
                    "future_ltp_5m": np.arange(20, dtype=float),
                }
            )
            pq = os.path.join(out_dir, f"{name}.parquet")
            js = os.path.join(out_dir, f"{name}.json")
            df.to_parquet(pq, index=False)
            with open(js, "w", encoding="utf-8") as fh:
                json.dump({"dataset_name": name, "dataset_kind": "unseen"}, fh)

            with self.assertRaises(ValueError) as ctx:
                load_unseen_xy(
                    data_dir=tmp,
                    dataset_name=name,
                    features=["f_ok", "f_missing_critical"],
                    target="future_ltp_5m",
                    unseen_days=["2026-07-30"],
                    parquet_path=pq,
                    json_path=js,
                )
            self.assertIn("missing", str(ctx.exception).lower())
            self.assertIn("f_missing_critical", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
