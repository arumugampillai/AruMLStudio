"""Self-contained experiment feature_set + details formatting."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.analysis_experiments import (
    STATUS_CREATED,
    STATUS_VALIDATED,
    build_feature_set,
    build_parent_diff,
    format_experiment_details,
    features_fingerprint,
)


class FeatureSetRecordTests(unittest.TestCase):
    def test_build_feature_set_hash_and_families(self) -> None:
        fs = build_feature_set(
            families=[
                {
                    "family_id": "f_price",
                    "family_label": "Price",
                    "representative": "option_ask",
                    "members": ["option_bid", "option_ask"],
                },
                {
                    "family_id": "f_iv",
                    "family_label": "IV",
                    "representative": "bs_reiv_pred",
                },
            ],
            discovery_bundle_id="DB-001",
            parent_experiment_id="Exp-001",
            variant_changes=[
                {
                    "family_id": "f_price",
                    "family_label": "Price",
                    "old_representative": "option_bid",
                    "new_representative": "option_ask",
                }
            ],
        )
        self.assertEqual(fs["count"], 2)
        self.assertEqual(fs["features"], ["option_ask", "bs_reiv_pred"])
        self.assertEqual(fs["hash"], features_fingerprint(fs["features"]))
        self.assertEqual(fs["changed_families"], ["Price"])
        self.assertTrue(fs["families"][0]["changed"])
        self.assertFalse(fs["families"][1]["changed"])

    def test_parent_diff_marks_changed(self) -> None:
        diff = build_parent_diff(
            parent_experiment_id="Exp-001",
            parent_family_reps=[
                {
                    "family_id": "f_price",
                    "family_label": "Price",
                    "representative": "option_bid",
                },
                {
                    "family_id": "f_iv",
                    "family_label": "IV",
                    "representative": "bs_reiv",
                },
            ],
            current_family_reps=[
                {
                    "family_id": "f_price",
                    "family_label": "Price",
                    "representative": "option_ask",
                },
                {
                    "family_id": "f_iv",
                    "family_label": "IV",
                    "representative": "bs_reiv_pred",
                },
            ],
        )
        self.assertEqual(diff["n_changed_families"], 2)
        by_id = {r["family_id"]: r for r in diff["rows"]}
        self.assertTrue(by_id["f_price"]["changed"])
        self.assertEqual(by_id["f_price"]["parent_representative"], "option_bid")
        self.assertEqual(by_id["f_price"]["representative"], "option_ask")

    def test_format_details_sections(self) -> None:
        exp = {
            "experiment_id": "Exp-150",
            "name": "swap-price",
            "status": STATUS_VALIDATED,
            "parent_experiment_id": "Exp-001",
            "discovery_bundle_id": "DB-20260729-001",
            "holdout_score": 0.82,
            "walk_forward_score": 0.80,
            "validation_label": "Good",
            "train_device": "GPU",
            "shap_device": "GPU",
            "device_label": "GPU",
            "model_name": "Exp_Exp_150",
            "result_artifact_id": "art-result",
            "is_champion": 0,
            "family_reps": [
                {
                    "family_id": "f_price",
                    "family_label": "Price",
                    "representative": "option_ask",
                },
                {
                    "family_id": "f_iv",
                    "family_label": "IV",
                    "representative": "bs_reiv_pred",
                },
            ],
            "variant_changes_list": [
                {
                    "family_id": "f_price",
                    "family_label": "Price",
                    "old_representative": "option_bid",
                    "new_representative": "option_ask",
                }
            ],
            "parent_family_reps": [
                {
                    "family_id": "f_price",
                    "family_label": "Price",
                    "representative": "option_bid",
                },
                {
                    "family_id": "f_iv",
                    "family_label": "IV",
                    "representative": "bs_reiv_pred",
                },
            ],
        }
        text = format_experiment_details(exp)
        self.assertIn("Experiment: Exp-150", text)
        self.assertIn("Final Feature Set", text)
        self.assertIn("1. option_ask", text)
        self.assertIn("Feature Set Summary", text)
        self.assertIn("Feature Family Summary", text)
        self.assertIn("Changes from Parent", text)
        self.assertIn("← Changed", text)
        self.assertIn("Model Information", text)
        self.assertIn("Train device       GPU", text)
        self.assertIn("Metrics", text)

    def test_created_shows_unchecked_status(self) -> None:
        text = format_experiment_details(
            {
                "experiment_id": "Exp-001",
                "status": STATUS_CREATED,
                "family_reps": [
                    {
                        "family_id": "f1",
                        "family_label": "Price",
                        "representative": "spot",
                    }
                ],
            }
        )
        self.assertIn("○ Trained", text)
        self.assertIn("○ Validated", text)
        self.assertIn("○ Champion", text)
        self.assertIn("1. spot", text)


if __name__ == "__main__":
    unittest.main()
