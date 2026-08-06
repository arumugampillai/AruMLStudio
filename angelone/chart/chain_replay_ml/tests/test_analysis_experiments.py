"""Tests for Experiment Manager — candidate-set experiments."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.analysis_correlation import (
    run_correlation_analysis,
)
from chain_replay_ml.dataset_builder.analysis_experiments import (
    VALIDATION_BEST,
    VALIDATION_WORSE,
    clone_experiment_with_rep,
    compare_experiments,
    create_experiment,
    list_experiments,
    load_experiment,
    update_experiment_metrics,
)
from chain_replay_ml.dataset_builder.analysis_family_review import (
    STATUS_FOR_EXPERIMENT,
    apply_discovery_suggestions,
    upsert_family_review,
)
from chain_replay_ml.dataset_builder.analysis_feature_profiles import (
    build_feature_profiles,
)
from chain_replay_ml.dataset_builder.analysis_hca import run_hca_analysis
from chain_replay_ml.dataset_builder.analysis_lab_store import (
    STATUS_COMPLETED,
    _AnalysisDb,
    _now_iso,
    ensure_analysis_run,
    register_dataset,
    set_module_status,
)


class ExperimentManagerTests(unittest.TestCase):
    def _setup(self, tmp: str) -> str:
        n = 80
        spot = pd.Series(np.linspace(100, 140, n))
        df = pd.DataFrame(
            {
                "trading_day": ["2024-01-02"] * n,
                "spot": spot,
                "spot_dup": spot * 1.00001,
                "microprice": spot + 0.01,
                "current_iv": np.linspace(0.1, 0.3, n),
                "noise_feat": np.random.default_rng(0).normal(size=n),
                "future_ltp_5m": spot.shift(-1).bfill(),
            }
        )
        path = os.path.join(tmp, "exp_demo.parquet")
        df.to_parquet(path, index=False)
        with open(path.replace(".parquet", ".json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "columns": list(df.columns),
                    "targets": ["future_ltp_5m"],
                    "labels": [],
                },
                f,
            )
        ds = register_dataset(tmp, path, name="exp_demo")
        run = ensure_analysis_run(tmp, ds["dataset_id"])
        run_id = run["run_id"]
        run_correlation_analysis(tmp, run_id, ds)
        build_feature_profiles(tmp, run_id, ds)
        run_hca_analysis(tmp, run_id)
        # Seed discovery scores
        now = _now_iso()
        with _AnalysisDb(tmp) as conn:
            for name, score in (
                ("spot", 81.0),
                ("spot_dup", 80.0),
                ("microprice", 79.0),
                ("current_iv", 70.0),
                ("noise_feat", 10.0),
            ):
                conn.execute(
                    """
                    UPDATE feature_profiles
                    SET feature_score = ?, rating_score = ?, updated_at = ?
                    WHERE run_id = ? AND feature_name = ?
                    """,
                    (score, score, now, run_id, name),
                )
        apply_discovery_suggestions(tmp, run_id)
        for mid in (
            "correlation",
            "hca",
            "mutual_information",
            "permutation",
            "feature_scorecard",
        ):
            set_module_status(tmp, run_id, mid, STATUS_COMPLETED, message="test")
        # Publish intermediate stage artifacts so discovery_bundle has parents
        from chain_replay_ml.dataset_builder.analysis_artifacts import (
            publish_module_artifact,
        )

        for mid in (
            "correlation",
            "hca",
            "mutual_information",
            "permutation",
            "feature_scorecard",
        ):
            publish_module_artifact(tmp, run_id, mid, summary={"test": True})
        return run_id

    def test_create_clone_compare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_id = self._setup(tmp)
            exp1 = create_experiment(tmp, run_id, name="baseline")
            self.assertEqual(exp1["experiment_id"], "Exp-001")
            self.assertTrue(exp1.get("family_reps"))
            self.assertTrue(exp1.get("discovery_bundle_id"))
            self.assertTrue(exp1.get("hypothesis_artifact_id"))

            # Mutating live Family Review must not change a frozen hypothesis
            from chain_replay_ml.dataset_builder.analysis_artifacts import (
                KIND_DISCOVERY_BUNDLE,
                require_artifact,
            )

            bundle = require_artifact(
                tmp,
                exp1["discovery_bundle_id"],
                expected_kind=KIND_DISCOVERY_BUNDLE,
            )
            frozen_reps = dict(bundle["payload"]["family_reps"])
            fam0 = exp1["family_reps"][0]
            fid = str(fam0["family_id"])
            cur = str(fam0["representative"])
            from chain_replay_ml.dataset_builder.analysis_hca import load_families

            members = []
            for f in load_families(tmp, run_id, min_size=2):
                if str(f.get("family_id")) == fid:
                    members = list(f.get("members") or [])
                    break
            alt = next((m for m in members if m != cur), None)
            if alt is None:
                self.skipTest("no alternate member for clone")

            upsert_family_review(
                tmp,
                run_id,
                fid,
                experiment_representative=alt,
                status=STATUS_FOR_EXPERIMENT,
                reason_code="Interpretability",
            )
            # Bundle payload unchanged
            bundle2 = require_artifact(
                tmp,
                exp1["discovery_bundle_id"],
                expected_kind=KIND_DISCOVERY_BUNDLE,
            )
            self.assertEqual(bundle2["payload"]["family_reps"], frozen_reps)
            exp2 = clone_experiment_with_rep(
                tmp,
                exp1["experiment_id"],
                family_id=fid,
                representative=alt,
            )
            self.assertEqual(exp2["experiment_id"], "Exp-002")
            self.assertEqual(exp2.get("parent_experiment_id"), "Exp-001")
            changes = exp2.get("variant_changes_list") or []
            self.assertEqual(len(changes), 1)
            self.assertEqual(changes[0].get("old_representative"), cur)
            self.assertEqual(changes[0].get("new_representative"), alt)

            # Identical snapshot rejected
            with self.assertRaises(ValueError):
                clone_experiment_with_rep(
                    tmp,
                    exp1["experiment_id"],
                    family_id=fid,
                    representative=cur,
                )

            update_experiment_metrics(
                tmp,
                exp1["experiment_id"],
                holdout_score=0.82,
                walk_forward_score=0.80,
                validation_label="Pending",
            )
            update_experiment_metrics(
                tmp,
                exp2["experiment_id"],
                holdout_score=0.84,
                walk_forward_score=0.83,
                validation_label="Pending",
            )

            rows = compare_experiments(tmp, run_id, focus_family_id=fid)
            self.assertEqual(len(rows), 2)
            by_id = {r["experiment_id"]: r for r in rows}
            self.assertEqual(by_id["Exp-002"]["focus_representative"], alt)
            self.assertEqual(by_id["Exp-002"]["validation_label"], VALIDATION_BEST)
            self.assertEqual(by_id["Exp-001"]["validation_label"], VALIDATION_WORSE)

            listed = list_experiments(tmp, run_id)
            self.assertEqual(len(listed), 2)

            from chain_replay_ml.dataset_builder.analysis_experiments import (
                STATUS_CHAMPION,
                STATUS_CREATED,
                STATUS_VALIDATED,
                discovery_bundle_card,
                platform_workflow_summary,
                promote_champion,
                request_train_experiment,
            )

            exp3 = create_experiment(
                tmp, run_id, name="train-me", family_reps={fid: alt}
            )
            self.assertEqual(exp3["status"], STATUS_CREATED)
            self.assertTrue(exp3.get("discovery_bundle_fingerprint"))
            loaded = load_experiment(
                tmp, exp3["experiment_id"], verify_bundle=True
            )
            assert loaded is not None
            self.assertTrue(loaded.get("fingerprint_ok"))
            self.assertIn("Final Feature Set", loaded.get("hypothesis_text") or "")
            self.assertIn("Feature Family Summary", loaded.get("hypothesis_text") or "")

            card = discovery_bundle_card(tmp, run_id)
            self.assertTrue(card.get("present"))
            self.assertEqual(card.get("artifact_id"), exp1["discovery_bundle_id"])
            self.assertGreaterEqual(int(card.get("n_experiments") or 0), 1)

            train = request_train_experiment(tmp, exp3["experiment_id"])
            self.assertEqual(train["status"], STATUS_VALIDATED)
            self.assertTrue(train["started"])
            self.assertTrue(train.get("result_artifact_id"))
            self.assertIsNotNone(train.get("holdout_r2"))
            trained = load_experiment(tmp, exp3["experiment_id"])
            assert trained is not None
            self.assertEqual(trained["status"], STATUS_VALIDATED)
            details = trained.get("hypothesis_text") or ""
            self.assertIn("Final Feature Set", details)
            self.assertIn("Model Information", details)
            self.assertIn("Metrics", details)
            self.assertIn(trained.get("train_device"), ("GPU", "CPU"))
            self.assertTrue(trained.get("device_label"))
            self.assertIn("Train device", details)
            fs = trained.get("feature_set") or {}
            self.assertGreaterEqual(int(fs.get("count") or 0), 1)
            self.assertTrue(fs.get("hash"))
            self.assertTrue(fs.get("features"))
            # Result artifact must carry self-contained feature_set
            from chain_replay_ml.dataset_builder.analysis_artifacts import (
                load_artifact,
            )

            result_art = load_artifact(tmp, trained["result_artifact_id"])
            assert result_art is not None
            rp = result_art.get("payload") or {}
            self.assertIn("feature_set", rp)
            selected = list(
                rp.get("selected_features") or rp.get("features") or []
            )
            model_names = list(rp.get("model_feature_names") or [])
            self.assertEqual(selected, list((rp.get("feature_set") or {}).get("features") or []))
            self.assertEqual(len(selected), len(model_names))
            self.assertEqual(selected, model_names)
            self.assertTrue(rp.get("feature_names_match"))
            self.assertIn("Model columns match", details)

            champ = promote_champion(tmp, exp3["experiment_id"])
            self.assertIn("champion_artifact_id", champ)
            self.assertTrue(str(champ.get("name") or "").startswith("Champion-"))
            self.assertIn("Selected Features", champ.get("card_text") or "")
            self.assertIn("Final Feature Set", champ.get("card_text") or "")
            self.assertGreaterEqual(int(champ.get("n_selected_features") or 0), 1)
            from chain_replay_ml.dataset_builder.analysis_experiments import (
                format_champion_bundle_card,
                load_champion_bundle,
            )

            bundle = load_champion_bundle(tmp, run_id)
            self.assertIsNotNone(bundle)
            assert bundle is not None
            self.assertEqual(
                bundle["artifact_id"], champ["champion_artifact_id"]
            )
            card_txt = format_champion_bundle_card(bundle)
            self.assertIn("Champion Bundle", card_txt)
            pl = bundle.get("payload") or {}
            self.assertIn(pl.get("train_device"), ("GPU", "CPU"))
            self.assertTrue(pl.get("device_label"))
            self.assertIn("feature_set", pl)
            self.assertTrue((pl.get("feature_set") or {}).get("features"))
            self.assertIn("Train device", card_txt)
            champ_exp = load_experiment(tmp, exp3["experiment_id"])
            assert champ_exp is not None
            self.assertEqual(int(champ_exp.get("is_champion") or 0), 1)
            self.assertEqual(champ_exp.get("status"), STATUS_CHAMPION)
            self.assertEqual(champ_exp.get("status_display"), STATUS_CHAMPION)

            wf = platform_workflow_summary(tmp, run_id)
            self.assertTrue(wf.get("discovery_complete"))
            self.assertEqual(wf.get("champion_id"), exp3["experiment_id"])
            self.assertGreaterEqual(int(wf.get("n_models") or 0), 1)
            self.assertIn("Champion", wf.get("banner_text") or "")

            from chain_replay_ml.dataset_builder.analysis_experiments import (
                delete_experiment,
            )

            # Validated/Champion cannot be deleted
            with self.assertRaises(ValueError):
                delete_experiment(tmp, exp3["experiment_id"])
            # Fresh Created snapshot can be deleted
            exp4 = create_experiment(tmp, run_id, name="to-delete")
            self.assertEqual(exp4["status"], STATUS_CREATED)
            deleted = delete_experiment(tmp, exp4["experiment_id"])
            self.assertTrue(deleted.get("deleted"))
            self.assertIsNone(load_experiment(tmp, exp4["experiment_id"]))


if __name__ == "__main__":
    unittest.main()
