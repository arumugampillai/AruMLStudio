"""Unit tests for Feature Recommendation Lifecycle store."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock


class RecommendationStrengthTests(unittest.TestCase):
    def test_strength_ladder(self) -> None:
        from chain_replay_ml.production_validation.recommendation_store import (
            compute_recommendation_strength,
            recommendation_strength_stars,
        )

        self.assertEqual(
            compute_recommendation_strength(
                remove_models=4, watch_models=0, keep_models=0
            ),
            5,
        )
        self.assertEqual(
            compute_recommendation_strength(
                remove_models=2, watch_models=0, keep_models=0
            ),
            4,
        )
        self.assertEqual(
            compute_recommendation_strength(
                remove_models=2, watch_models=1, keep_models=1
            ),
            3,
        )
        self.assertEqual(
            compute_recommendation_strength(
                remove_models=0, watch_models=3, keep_models=1
            ),
            2,
        )
        self.assertEqual(
            compute_recommendation_strength(
                remove_models=0, watch_models=0, keep_models=3
            ),
            1,
        )
        self.assertEqual(recommendation_strength_stars(5), "★★★★★")
        self.assertEqual(recommendation_strength_stars(1), "★☆☆☆☆")


class RecommendationStoreTests(unittest.TestCase):
    def _write_pv_artifacts(
        self,
        data_dir: str,
        model_name: str,
        rows: list[dict],
        *,
        run_id: str = "run-abc",
    ) -> str:
        from chain_replay_ml.training.paths import model_package_dir, safe_model_name

        pkg = model_package_dir(data_dir, safe_model_name(model_name))
        out = os.path.join(pkg, "production_validation")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "comparison.json"), "w", encoding="utf-8") as fh:
            json.dump({"rows": rows}, fh)
        with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump({"feature_validation": {}}, fh)
        with open(os.path.join(out, "run_meta.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "run_id": run_id,
                    "generated_at": "2026-08-04T10:00:00+00:00",
                    "model_name": safe_model_name(model_name),
                },
                fh,
            )
        return pkg

    def test_unique_models_not_raw_runs(self) -> None:
        from chain_replay_ml.production_validation.recommendation_store import (
            get_recommendation_summary,
            update_registry_recommendations,
        )

        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                {"feature": "weighted_ltp_ema", "recommendation": "REMOVE"},
                {"feature": "iv_change", "recommendation": "KEEP"},
            ]
            self._write_pv_artifacts(tmp, "ModelA", rows, run_id="run-1")
            update_registry_recommendations(tmp, model_name="ModelA")
            # Same model, new run → runs++ but models stay 1 for REMOVE
            self._write_pv_artifacts(tmp, "ModelA", rows, run_id="run-2")
            update_registry_recommendations(tmp, model_name="ModelA")
            # Different model → models++
            self._write_pv_artifacts(tmp, "ModelB", rows, run_id="run-3")
            update_registry_recommendations(tmp, model_name="ModelB")

            summary = get_recommendation_summary(tmp)
            by_name = {f["feature_name"]: f for f in summary["features"]}
            ema = by_name["weighted_ltp_ema"]
            self.assertEqual(ema["remove_runs"], 3)
            self.assertEqual(ema["remove_models"], 2)
            self.assertEqual(ema["keep_runs"], 0)
            iv = by_name["iv_change"]
            self.assertEqual(iv["keep_runs"], 3)
            self.assertEqual(iv["keep_models"], 2)
            self.assertGreaterEqual(ema["recommendation_strength"], 3)

    def test_idempotent_same_run(self) -> None:
        from chain_replay_ml.production_validation.recommendation_store import (
            list_recommendation_history,
            update_registry_recommendations,
        )

        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                {"feature": "spot_ratio", "recommendation": "REMOVE"},
                {"feature": "keep_me", "recommendation": "KEEP"},
            ]
            self._write_pv_artifacts(tmp, "ModelX", rows, run_id="same-run")
            r1 = update_registry_recommendations(tmp, model_name="ModelX")
            r2 = update_registry_recommendations(tmp, model_name="ModelX")
            self.assertEqual(r1["inserted"], 2)
            self.assertEqual(r2["inserted"], 0)
            self.assertEqual(r2["updated"], 2)
            hist = list_recommendation_history(tmp, feature_name="spot_ratio")
            self.assertEqual(len(hist), 1)
            self.assertEqual(hist[0]["production_validation_run_id"], "same-run")

    def test_remove_only_filter(self) -> None:
        from chain_replay_ml.production_validation.recommendation_store import (
            list_recommendation_history,
            update_registry_recommendations,
        )

        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                {"feature": "a", "recommendation": "REMOVE"},
                {"feature": "b", "recommendation": "WATCH"},
                {"feature": "c", "recommendation": "KEEP"},
            ]
            self._write_pv_artifacts(tmp, "ModelY", rows)
            out = update_registry_recommendations(
                tmp,
                model_name="ModelY",
                recommendations={"REMOVE"},
            )
            self.assertEqual(out["inserted"], 1)
            self.assertEqual(len(list_recommendation_history(tmp)), 1)
            self.assertEqual(
                list_recommendation_history(tmp)[0]["recommendation"], "REMOVE"
            )

    def test_ignore_and_removal_list(self) -> None:
        from chain_replay_ml.production_validation.recommendation_store import (
            ignore_recommendation,
            recommended_for_removal,
            unignore_recommendation,
            update_registry_recommendations,
        )

        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                {"feature": "gone", "recommendation": "REMOVE"},
                {"feature": "stay", "recommendation": "KEEP"},
            ]
            self._write_pv_artifacts(tmp, "ModelZ", rows)
            update_registry_recommendations(tmp, model_name="ModelZ")
            removal = recommended_for_removal(tmp)
            self.assertEqual([r["feature_name"] for r in removal], ["gone"])
            ignore_recommendation(tmp, "gone", scope="pipeline")
            self.assertEqual(recommended_for_removal(tmp), [])
            self.assertEqual(
                len(recommended_for_removal(tmp, include_ignored=True)), 1
            )
            self.assertTrue(unignore_recommendation(tmp, "gone"))
            self.assertEqual(
                [r["feature_name"] for r in recommended_for_removal(tmp)], ["gone"]
            )

    def test_missing_artifacts_raises(self) -> None:
        from chain_replay_ml.production_validation.recommendation_store import (
            update_registry_recommendations,
        )

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                update_registry_recommendations(tmp, model_name="NoPV")

    def test_assigns_run_id_when_missing(self) -> None:
        from chain_replay_ml.production_validation.recommendation_store import (
            update_registry_recommendations,
        )
        from chain_replay_ml.training.paths import model_package_dir, safe_model_name

        with tempfile.TemporaryDirectory() as tmp:
            pkg = model_package_dir(tmp, safe_model_name("Legacy"))
            out = os.path.join(pkg, "production_validation")
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "comparison.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "rows": [
                            {"feature": "f1", "recommendation": "REMOVE"},
                        ]
                    },
                    fh,
                )
            with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
                json.dump({}, fh)
            with open(os.path.join(out, "run_meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"model_name": "Legacy"}, fh)

            result = update_registry_recommendations(tmp, model_name="Legacy")
            self.assertTrue(result["production_validation_run_id"])
            with open(os.path.join(out, "run_meta.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertEqual(meta["run_id"], result["production_validation_run_id"])

    def test_feature_id_resolved_from_registry_store(self) -> None:
        from chain_replay_ml.production_validation.recommendation_store import (
            list_recommendation_history,
            update_registry_recommendations,
        )

        with tempfile.TemporaryDirectory() as tmp:
            self._write_pv_artifacts(
                tmp,
                "ModelId",
                [{"feature": "weighted_ltp_ema", "recommendation": "REMOVE"}],
            )
            with mock.patch(
                "chain_replay_ml.production_validation.recommendation_store._feature_id_map",
                return_value={"weighted_ltp_ema": "FR0042"},
            ):
                update_registry_recommendations(tmp, model_name="ModelId")
            hist = list_recommendation_history(tmp)
            self.assertEqual(hist[0]["feature_id"], "FR0042")

    def test_persist_api_wrapper(self) -> None:
        from chain_replay_ml.production_validation import persist_registry_recommendations

        with tempfile.TemporaryDirectory() as tmp:
            self._write_pv_artifacts(
                tmp,
                "Wrap",
                [{"feature": "x", "recommendation": "WATCH"}],
            )
            out = persist_registry_recommendations(data_dir=tmp, model_name="Wrap")
            self.assertTrue(out["ok"])
            self.assertEqual(out["inserted"], 1)


if __name__ == "__main__":
    unittest.main()
