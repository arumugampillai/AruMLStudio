"""Tests for Feature Rating — Discovery has no SHAP; Validation is separate."""

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
from chain_replay_ml.dataset_builder.analysis_feature_profiles import (
    build_feature_profiles,
    load_feature_profile,
    load_feature_scorecard,
)
from chain_replay_ml.dataset_builder.analysis_feature_rating import (
    ACTION_KEEP,
    ACTION_MERGE,
    ACTION_RETIRE,
    ACTION_REVIEW,
    STAGE_DISCOVERY,
    STAGE_VALIDATION,
    VAL_PRODUCTION_READY,
    decide_action,
    decide_discovery_action,
    format_score_cell,
    load_feature_ratings,
    rating_stars,
    run_feature_rating,
)
from chain_replay_ml.dataset_builder.analysis_lab_store import (
    MODULE_DEPENDS_ON,
    STATUS_COMPLETED,
    ensure_analysis_run,
    module_statuses,
    register_dataset,
)
from chain_replay_ml.dataset_builder.analysis_mutual_information import (
    run_mutual_information,
)
from chain_replay_ml.dataset_builder.analysis_permutation import (
    run_permutation_importance,
)
from chain_replay_ml.dataset_builder.analysis_shap import run_shap_analysis
from chain_replay_ml.training.paths import model_package_dir


def _train_tiny_xgb(tmp: str, features: list[str], model_name: str) -> None:
    from xgboost import XGBRegressor

    rng = np.random.default_rng(0)
    n = 120
    X = {f: rng.normal(size=n) for f in features}
    y = X[features[0]] * 3.0 + rng.normal(scale=0.05, size=n)
    model = XGBRegressor(
        n_estimators=12,
        max_depth=2,
        learning_rate=0.3,
        objective="reg:squarederror",
        verbosity=0,
    )
    model.fit(pd.DataFrame(X), y)
    pkg = model_package_dir(tmp, model_name)
    os.makedirs(pkg, exist_ok=True)
    model.save_model(os.path.join(pkg, "model.ubj"))
    with open(os.path.join(pkg, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_features": features,
                "features": features,
                "algorithm": "xgboost",
                "target": "future_ltp_5m",
            },
            f,
        )
    with open(os.path.join(pkg, "registry.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "algorithm": "xgboost",
                "target": "future_ltp_5m",
                "status": "trained",
                "dataset": "rating_demo",
            },
            f,
        )


class DecideActionTests(unittest.TestCase):
    def test_discovery_keep_ignores_shap_args(self) -> None:
        action, conf, reason = decide_action(
            mi_pct=95.0,
            shap_pct=1.0,
            perm_pct=88.0,
            abs_corr=0.40,
            shap_share=0.0,
            perm_mean=0.5,
            coverage=99.0,
            stage=STAGE_DISCOVERY,
        )
        self.assertEqual(action, ACTION_KEEP)
        self.assertNotIn("SHAP", reason)
        self.assertNotIn("shap", reason.lower())

    def test_discovery_decide_has_no_shap_params(self) -> None:
        action, conf, reason = decide_discovery_action(
            mi_pct=40.0,
            perm_pct=5.0,
            abs_corr=0.9999,
            perm_mean=0.02,
            coverage=99.0,
        )
        self.assertEqual(action, ACTION_MERGE)
        self.assertNotIn("SHAP", reason)

    def test_discovery_retire_without_shap(self) -> None:
        action, conf, reason = decide_discovery_action(
            mi_pct=5.0,
            perm_pct=1.0,
            abs_corr=0.20,
            perm_mean=0.0,
            coverage=98.0,
        )
        self.assertEqual(action, ACTION_RETIRE)
        self.assertNotIn("SHAP", reason)

    def test_discovery_review_family_reason(self) -> None:
        action, conf, reason = decide_discovery_action(
            mi_pct=50.0,
            perm_pct=50.0,
            abs_corr=0.90,
            perm_mean=0.1,
            coverage=99.0,
        )
        self.assertEqual(action, ACTION_REVIEW)
        self.assertIn("HCA family", reason)
        self.assertNotIn("SHAP", reason)

    def test_validation_keep_mentions_shap(self) -> None:
        action, conf, reason = decide_action(
            mi_pct=95.0,
            shap_pct=92.0,
            perm_pct=88.0,
            abs_corr=0.40,
            shap_share=0.12,
            perm_mean=0.5,
            coverage=99.0,
            stage=STAGE_VALIDATION,
        )
        self.assertEqual(action, VAL_PRODUCTION_READY)
        self.assertIn("SHAP", reason)

    def test_rating_stars_and_score_cell(self) -> None:
        self.assertEqual(rating_stars(97), "★★★★★")
        self.assertEqual(format_score_cell({}), "Pending")

    def test_feature_rating_deps_exclude_shap(self) -> None:
        deps = MODULE_DEPENDS_ON.get("feature_scorecard") or ()
        self.assertNotIn("shap", deps)


class FeatureRatingIntegrationTests(unittest.TestCase):
    def _env(self, tmp: str) -> tuple[str, dict, str]:
        features = ["spot", "current_iv", "noise_feat"]
        n = 100
        rng = np.random.default_rng(11)
        spot = pd.Series(np.linspace(100, 140, n)) + rng.normal(0, 0.2, n)
        spot_dup = spot * 1.00001 + 1e-9
        df = pd.DataFrame(
            {
                "trading_day": ["2024-01-02"] * n,
                "spot": spot,
                "spot_dup": spot_dup,
                "current_iv": rng.uniform(0.1, 0.4, n),
                "noise_feat": rng.normal(size=n),
                "future_ltp_5m": spot.shift(-1).bfill() + rng.normal(0, 0.05, n),
            }
        )
        path = os.path.join(tmp, "analysis_rating_demo.parquet")
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
        ds = register_dataset(tmp, path, name="analysis_rating_demo")
        run = ensure_analysis_run(tmp, ds["dataset_id"])
        model_name = "rating_tiny_xgb"
        _train_tiny_xgb(tmp, features, model_name)
        return run["run_id"], ds, model_name

    def test_discovery_rating_never_mentions_shap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_id, ds, model_name = self._env(tmp)
            run_correlation_analysis(tmp, run_id, ds)
            build_feature_profiles(tmp, run_id, ds)
            run_mutual_information(tmp, run_id, ds, "future_ltp_5m")
            run_permutation_importance(
                tmp, run_id, ds, model_name, "future_ltp_5m"
            )

            out = run_feature_rating(
                tmp,
                run_id,
                model_name=model_name,
                target="future_ltp_5m",
                stage=STAGE_DISCOVERY,
            )
            self.assertEqual(out["stage"], STAGE_DISCOVERY)
            self.assertFalse(out["modules_used"]["shap"])

            statuses = {
                r["module_id"]: r["status"] for r in module_statuses(tmp, run_id)
            }
            self.assertEqual(statuses.get("feature_scorecard"), STATUS_COMPLETED)

            for r in load_feature_scorecard(tmp, run_id):
                for key in ("rating_reason", "reason", "recommendation", "rating_action"):
                    self.assertNotIn("SHAP", str(r.get(key) or ""))
                    self.assertNotIn("shap", str(r.get(key) or "").lower())

            for r in load_feature_ratings(tmp, run_id):
                self.assertNotIn("SHAP", str(r.get("rating_reason") or ""))

    def test_validation_does_not_overwrite_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_id, ds, model_name = self._env(tmp)
            run_correlation_analysis(tmp, run_id, ds)
            build_feature_profiles(tmp, run_id, ds)
            run_mutual_information(tmp, run_id, ds, "future_ltp_5m")
            run_permutation_importance(
                tmp, run_id, ds, model_name, "future_ltp_5m"
            )
            run_feature_rating(
                tmp,
                run_id,
                model_name=model_name,
                target="future_ltp_5m",
                stage=STAGE_DISCOVERY,
            )
            before = {
                r["feature_name"]: (
                    r.get("rating_action"),
                    r.get("rating_reason"),
                    r.get("feature_score"),
                )
                for r in load_feature_scorecard(tmp, run_id)
            }

            run_shap_analysis(tmp, run_id, ds, model_name)
            out = run_feature_rating(
                tmp,
                run_id,
                model_name=model_name,
                target="future_ltp_5m",
                stage=STAGE_VALIDATION,
            )
            self.assertEqual(out["stage"], STAGE_VALIDATION)

            after = load_feature_scorecard(tmp, run_id)
            for r in after:
                name = r["feature_name"]
                self.assertEqual(r.get("rating_action"), before[name][0])
                self.assertEqual(r.get("rating_reason"), before[name][1])
                self.assertEqual(r.get("feature_score"), before[name][2])
                self.assertNotIn("SHAP", str(r.get("rating_reason") or ""))
                # Validation lives in separate columns
                if r.get("validation_reason"):
                    self.assertIn("SHAP", str(r.get("validation_reason")))


if __name__ == "__main__":
    unittest.main()
