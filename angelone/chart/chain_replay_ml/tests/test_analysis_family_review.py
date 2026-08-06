"""Tests for Family Review exception queue + Discovery suggestions."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.analysis_correlation import (
    run_correlation_analysis,
)
from chain_replay_ml.dataset_builder.analysis_family_review import (
    FILTER_ALL,
    FILTER_NEEDS_REVIEW,
    GAP_HIGH,
    STATUS_ACCEPTED,
    STATUS_AUTO_ACCEPTED,
    STATUS_DECISION_REQUIRED,
    STATUS_REVIEW_RECOMMENDED,
    apply_discovery_suggestions,
    confidence_from_gap,
    load_families_with_reviews,
    load_family_review,
    review_summary,
    status_from_confidence,
    suggest_family_representative,
    upsert_family_review,
)
from chain_replay_ml.dataset_builder.analysis_feature_profiles import (
    build_feature_profiles,
)
from chain_replay_ml.dataset_builder.analysis_hca import run_hca_analysis
from chain_replay_ml.dataset_builder.analysis_lab_store import (
    _AnalysisDb,
    _now_iso,
    ensure_analysis_run,
    register_dataset,
)
from chain_replay_ml.dataset_builder.analysis_mutual_information import (
    run_mutual_information,
)
from chain_replay_ml.dataset_builder.analysis_permutation import (
    run_permutation_importance,
)
from chain_replay_ml.training.paths import model_package_dir
import json


class ConfidenceLogicTests(unittest.TestCase):
    def test_gap_bands(self) -> None:
        self.assertEqual(confidence_from_gap(20), "High")
        self.assertEqual(confidence_from_gap(GAP_HIGH), "High")
        self.assertEqual(confidence_from_gap(8), "Medium")
        self.assertEqual(confidence_from_gap(2), "Low")
        self.assertEqual(status_from_confidence("High"), STATUS_AUTO_ACCEPTED)
        self.assertEqual(
            status_from_confidence("Medium"), STATUS_REVIEW_RECOMMENDED
        )
        self.assertEqual(
            status_from_confidence("Low"), STATUS_DECISION_REQUIRED
        )

    def test_suggest_clear_winner(self) -> None:
        sug = suggest_family_representative(
            ["a", "b", "c"],
            {"a": 96.0, "b": 74.0, "c": 50.0},
        )
        self.assertEqual(sug["suggested_representative"], "a")
        self.assertEqual(sug["confidence"], "High")
        self.assertEqual(sug["status"], STATUS_AUTO_ACCEPTED)
        self.assertGreaterEqual(float(sug["score_gap"]), GAP_HIGH)

    def test_suggest_near_tie(self) -> None:
        sug = suggest_family_representative(
            ["microprice", "mid_price", "ltp"],
            {"microprice": 91.0, "mid_price": 90.0, "ltp": 89.0},
        )
        self.assertEqual(sug["suggested_representative"], "microprice")
        self.assertEqual(sug["confidence"], "Low")
        self.assertEqual(sug["status"], STATUS_DECISION_REQUIRED)


class ExceptionQueueIntegrationTests(unittest.TestCase):
    def _train_tiny(self, tmp: str, features: list[str], name: str) -> None:
        from xgboost import XGBRegressor

        rng = np.random.default_rng(0)
        n = 80
        X = {f: rng.normal(size=n) for f in features}
        y = X[features[0]] * 2 + rng.normal(scale=0.1, size=n)
        model = XGBRegressor(
            n_estimators=8, max_depth=2, learning_rate=0.4, verbosity=0
        )
        model.fit(pd.DataFrame(X), y)
        pkg = model_package_dir(tmp, name)
        os.makedirs(pkg, exist_ok=True)
        model.save_model(os.path.join(pkg, "model.ubj"))
        with open(os.path.join(pkg, "config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "selected_features": features,
                    "algorithm": "xgboost",
                    "target": "future_ltp_5m",
                },
                f,
            )
        with open(os.path.join(pkg, "registry.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": name,
                    "algorithm": "xgboost",
                    "target": "future_ltp_5m",
                    "status": "trained",
                },
                f,
            )

    def test_apply_suggestions_and_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            n = 100
            rng = np.random.default_rng(5)
            spot = pd.Series(np.linspace(100, 140, n)) + rng.normal(0, 0.1, n)
            df = pd.DataFrame(
                {
                    "trading_day": ["2026-01-02"] * n,
                    "spot": spot,
                    "ltp": spot * 1.0001,
                    "mid_price": spot * 1.00005,
                    "current_iv": np.linspace(0.1, 0.25, n),
                    "iv_ema": np.linspace(0.11, 0.26, n),
                    "noise_feat": rng.normal(size=n),
                    "future_ltp_5m": spot.shift(-1).bfill(),
                }
            )
            path = os.path.join(tmp, "exc_demo.parquet")
            df.to_parquet(path, index=False)
            ds = register_dataset(tmp, path, name="exc_demo")
            run = ensure_analysis_run(tmp, ds["dataset_id"])
            run_id = run["run_id"]
            feats = ["spot", "ltp", "mid_price", "current_iv", "iv_ema", "noise_feat"]
            self._train_tiny(tmp, feats, "exc_xgb")

            run_correlation_analysis(tmp, run_id, ds)
            build_feature_profiles(tmp, run_id, ds)
            run_hca_analysis(tmp, run_id)
            run_mutual_information(tmp, run_id, ds, "future_ltp_5m")
            run_permutation_importance(
                tmp, run_id, ds, "exc_xgb", "future_ltp_5m"
            )

            # Seed discovery scores: one clear family winner, one near-tie family
            with _AnalysisDb(tmp) as conn:
                now = _now_iso()
                # Price-like: clear gap
                for name, score in (
                    ("spot", 96.0),
                    ("ltp", 70.0),
                    ("mid_price", 68.0),
                ):
                    conn.execute(
                        """
                        UPDATE feature_profiles
                        SET feature_score = ?, rating_score = ?, updated_at = ?
                        WHERE run_id = ? AND feature_name = ?
                        """,
                        (score, score, now, run_id, name),
                    )
                # IV-like: near tie
                for name, score in (
                    ("current_iv", 91.0),
                    ("iv_ema", 90.0),
                ):
                    conn.execute(
                        """
                        UPDATE feature_profiles
                        SET feature_score = ?, rating_score = ?, updated_at = ?
                        WHERE run_id = ? AND feature_name = ?
                        """,
                        (score, score, now, run_id, name),
                    )
                conn.execute(
                    """
                    UPDATE feature_profiles
                    SET feature_score = ?, rating_score = ?, updated_at = ?
                    WHERE run_id = ? AND feature_name = ?
                    """,
                    (10.0, 10.0, now, run_id, "noise_feat"),
                )

            out = apply_discovery_suggestions(tmp, run_id)
            self.assertGreater(out["n_families"], 0)
            self.assertGreaterEqual(
                out["n_auto_accepted"] + out["n_needs_review"], out["n_families"]
            )

            summary = review_summary(tmp, run_id, min_size=2)
            needs = load_families_with_reviews(
                tmp, run_id, min_size=2, status_filter=FILTER_NEEDS_REVIEW
            )
            all_rows = load_families_with_reviews(
                tmp, run_id, min_size=2, status_filter=FILTER_ALL
            )
            self.assertEqual(len(all_rows), summary["n_families"])
            self.assertEqual(len(needs), summary["n_needs_review"])
            # Default exception queue is smaller than or equal to all
            self.assertLessEqual(len(needs), len(all_rows))

            # Override a needs-review family if any exist; else override auto
            target = needs[0] if needs else all_rows[0]
            fid = str(target["family_id"])
            members = list(target["members"])
            pick = members[-1]
            upsert_family_review(
                tmp,
                run_id,
                fid,
                selected_representative=pick,
                reason_code="Interpretability",
                reason_text="Prefer clearer feature",
                status=STATUS_ACCEPTED,
            )
            loaded = load_family_review(tmp, run_id, fid)
            assert loaded is not None
            self.assertEqual(loaded["selected_representative"], pick)
            self.assertEqual(loaded["decision_source"], "manual")
            self.assertEqual(loaded["status"], STATUS_ACCEPTED)
            self.assertEqual(loaded["experiment_representative"], pick)

            from chain_replay_ml.dataset_builder.analysis_family_review import (
                discovery_readiness,
            )

            ready = discovery_readiness(tmp, run_id)
            self.assertIn("banner_text", ready)
            self.assertGreaterEqual(ready["n_families"], 1)
            self.assertIn("n_assigned", ready)
            self.assertIn("complete", ready)

            # Re-apply suggestions must keep manual decision
            out2 = apply_discovery_suggestions(tmp, run_id)
            self.assertGreaterEqual(out2.get("n_skipped_manual", 0), 1)
            loaded2 = load_family_review(tmp, run_id, fid)
            assert loaded2 is not None
            self.assertEqual(loaded2["selected_representative"], pick)
            self.assertEqual(loaded2["status"], STATUS_ACCEPTED)

            from chain_replay_ml.dataset_builder.analysis_family_review import (
                format_family_context_text,
                sync_scorecard_family_links,
            )
            from chain_replay_ml.dataset_builder.analysis_feature_rating import (
                ACTION_REVIEW,
            )

            link = sync_scorecard_family_links(tmp, run_id)
            self.assertGreaterEqual(link["n_features_linked"], 1)
            ctx = format_family_context_text(tmp, run_id, members[0])
            self.assertIn("Family", ctx)
            self.assertIn("Discovery Scores", ctx)
            # Needs-review members should carry REVIEW FAMILY on profiles
            if link["n_review_family"] > 0:
                with _AnalysisDb(tmp) as conn:
                    row = conn.execute(
                        """
                        SELECT rating_action, rating_family_id, rating_reason
                        FROM feature_profiles
                        WHERE run_id = ? AND rating_action = ?
                        LIMIT 1
                        """,
                        (run_id, ACTION_REVIEW),
                    ).fetchone()
                self.assertIsNotNone(row)
                self.assertTrue(row["rating_family_id"])
                self.assertIn("Open Family Review", str(row["rating_reason"] or ""))


if __name__ == "__main__":
    unittest.main()
