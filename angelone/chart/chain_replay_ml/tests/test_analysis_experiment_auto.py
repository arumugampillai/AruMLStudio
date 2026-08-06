"""Tests for Auto Create & Train (baseline + intelligent variants)."""

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
from chain_replay_ml.dataset_builder.analysis_experiment_auto import (
    STRATEGY_GREEDY,
    STRATEGY_HILL,
    STRATEGY_SINGLE_SWAP,
    auto_create_and_train,
    build_research_statistics,
    estimate_search_space,
    format_champion_card,
    format_dashboard,
    format_research_history,
    format_research_statistics,
    propose_intelligent_variants,
    recommend_champion,
)
from chain_replay_ml.dataset_builder.analysis_family_review import (
    apply_discovery_suggestions,
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


class AutoCreateTrainTests(unittest.TestCase):
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
        path = os.path.join(tmp, "auto_demo.parquet")
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
        ds = register_dataset(tmp, path, name="auto_demo")
        run = ensure_analysis_run(tmp, ds["dataset_id"])
        run_id = run["run_id"]
        run_correlation_analysis(tmp, run_id, ds)
        build_feature_profiles(tmp, run_id, ds)
        run_hca_analysis(tmp, run_id)
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
        from chain_replay_ml.dataset_builder.analysis_artifacts import (
            publish_module_artifact,
            publish_discovery_bundle,
        )

        for mid in (
            "correlation",
            "hca",
            "mutual_information",
            "permutation",
            "feature_scorecard",
        ):
            publish_module_artifact(tmp, run_id, mid, summary={"test": True})
        publish_discovery_bundle(tmp, run_id)
        return run_id

    def test_propose_and_auto_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_id = self._setup(tmp)
            proposals = propose_intelligent_variants(
                tmp, run_id, max_variants=9
            )
            self.assertGreaterEqual(len(proposals), 1)
            for p in proposals:
                self.assertNotEqual(
                    p["old_representative"], p["new_representative"]
                )
                self.assertEqual(int(p["rank_to"]), int(p["rank_from"]) + 1)

            progress: list[str] = []

            def on_progress(_frac: float, msg: str) -> None:
                progress.append(msg)

            out = auto_create_and_train(
                tmp,
                run_id,
                strategy=STRATEGY_GREEDY,
                max_variants=2,
                max_rounds=2,
                min_improvement=0.001,
                promote=True,
                on_progress=on_progress,
            )
            self.assertTrue(out.get("baseline_experiment_id"))
            self.assertLessEqual(len(out.get("variant_experiment_ids") or []), 4)
            self.assertTrue(out.get("recommended_champion_id"))
            self.assertTrue(out.get("recommendation"))
            self.assertIn("Current Best", out["recommendation"].get("text") or "")
            self.assertIn("Next Action", out["recommendation"].get("text") or "")
            self.assertEqual(out.get("strategy"), STRATEGY_GREEDY)
            self.assertEqual(out.get("strategy_label"), "Greedy Search")
            self.assertTrue(out.get("dashboard"))
            self.assertTrue(out.get("dashboard_text"))
            self.assertIn("Auto Research", out["dashboard_text"])
            self.assertIn("Search State", out["dashboard_text"])
            self.assertIn("Research Level", out["dashboard_text"])
            self.assertIn("Balanced", out["dashboard_text"])
            self.assertTrue(out.get("champion_card"))
            self.assertIn("Champion Feature Set", out["champion_card"])
            self.assertIn("Estimated Search Space", out["dashboard_text"])
            self.assertTrue(out.get("research_statistics"))
            self.assertIn(
                "Improvement over Baseline",
                out.get("research_statistics_text") or "",
            )
            self.assertIn("Research History", out["dashboard_text"])
            self.assertTrue(out.get("research_history"))
            self.assertEqual(out["research_history"][0].get("round"), 0)
            self.assertIn("Champion Score", out["dashboard_text"])
            dash = out["dashboard"]
            self.assertIsNotNone(dash.get("n_families_total"))
            self.assertGreaterEqual(int(dash.get("families_tested") or 0), 0)
            self.assertTrue(progress)
            self.assertTrue(out.get("features_fingerprints"))
            champ = recommend_champion(
                [e for e in (out.get("experiments") or []) if e]
            )
            self.assertIsNotNone(champ)
            self.assertEqual(
                champ["experiment_id"], out["recommended_champion_id"]
            )
            promote = out.get("promote") or {}
            self.assertTrue(promote.get("champion_artifact_id") or promote.get("error"))
            if promote.get("name"):
                self.assertTrue(str(promote["name"]).startswith("Champion-"))
            if promote.get("card_text"):
                self.assertIn("Champion Bundle", promote["card_text"])
                self.assertIn("Selected Features", promote["card_text"])

            # Hill climbing
            out_hill = auto_create_and_train(
                tmp,
                run_id,
                strategy=STRATEGY_HILL,
                max_variants=2,
                max_rounds=2,
                min_improvement=0.001,
                promote=False,
            )
            self.assertEqual(out_hill.get("strategy"), STRATEGY_HILL)
            self.assertIn("Search State", out_hill.get("dashboard_text") or "")
            self.assertTrue(out_hill.get("research_history"))
            self.assertIn(
                "Converged?", out_hill.get("research_statistics_text") or ""
            )

    def test_single_swap_forces_one_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_id = self._setup(tmp)
            out_ss = auto_create_and_train(
                tmp,
                run_id,
                strategy=STRATEGY_SINGLE_SWAP,
                max_variants=2,
                max_rounds=9,
                promote=False,
            )
            self.assertEqual(out_ss.get("strategy"), STRATEGY_SINGLE_SWAP)
            self.assertEqual(int(out_ss.get("max_rounds") or 0), 1)
            self.assertEqual(int(out_ss.get("rounds_done") or 0), 1)
            self.assertFalse(out_ss.get("can_continue"))
            self.assertTrue(out_ss.get("champion_card"))
            self.assertIn("Single Swap", out_ss.get("dashboard_text") or "")
            # Intent label for single_swap
            self.assertIn("Quick", out_ss.get("dashboard_text") or "")

    def test_format_dashboard_coverage_and_champion_card(self) -> None:
        text = format_dashboard(
            {
                "round": 1,
                "max_rounds": 3,
                "strategy_label": "Single Swap",
                "current_baseline_id": "Exp-001",
                "current_score_txt": "0.99000",
                "best_ever_id": "Exp-002",
                "best_ever_score_txt": "0.99100",
                "families_improved_txt": "IV",
                "families_changed": 1,
                "remaining_families": 46,
                "families_tested": 11,
                "n_families_total": 57,
                "status": "complete",
            }
        )
        self.assertIn("Research Coverage", text)
        self.assertIn("11 / 57", text)
        self.assertIn("19%", text)
        self.assertIn("Research Level", text)
        self.assertIn("Quick", text)
        card = format_champion_card(
            experiment={
                "experiment_id": "Exp-025",
                "holdout_score": 0.99135,
                "walk_forward_score": 0.99195,
                "validation_label": "Excellent",
                "family_reps": [
                    {"family_id": "iv", "representative": "iv_a"},
                ],
            },
            baseline_reps={"iv": "iv_b"},
            fam_labels={"iv": "IV Family"},
            stop_reason="No improvement found after Round 1",
            overall_score=94.0,
            research_complete=True,
        )
        self.assertIn("Exp-025", card)
        self.assertIn("READY FOR PRODUCTION", card)
        self.assertIn("✓ IV Family", card)
        self.assertIn("94/100", card)

        stats = build_research_statistics(
            run_id="analysis_206r",
            dataset="analysis_206r…",
            n_features=425,
            n_families=57,
            n_experiments=95,
            n_models=95,
            rounds=2,
            strategy_label="Hill Climbing",
            champion_id="Exp-150",
            champion_holdout=0.99272,
            champion_walk_forward=0.99195,
            baseline_holdout=0.99041,
            research_complete=False,
            neighbours_evaluated=234,
            best_improvement=0.00231,
            last_improvement=0.00012,
            converged=False,
            search_space=estimate_search_space(
                {
                    "families": [
                        {"members": ["a", "b", "c", "d"]} for _ in range(57)
                    ]
                }
            ),
            research_history=[
                {
                    "round": 0,
                    "champion_id": "Exp-001",
                    "score": 0.99041,
                    "last_improvement": None,
                },
                {
                    "round": 1,
                    "champion_id": "Exp-025",
                    "score": 0.99134,
                    "last_improvement": 0.00093,
                },
                {
                    "round": 2,
                    "champion_id": "Exp-150",
                    "score": 0.99272,
                    "last_improvement": 0.00138,
                },
            ],
        )
        stats_txt = format_research_statistics(stats)
        self.assertIn("Features                   425", stats_txt)
        self.assertIn("Experiments                95", stats_txt)
        self.assertIn("Exp-150", stats_txt)
        self.assertIn("Search State", stats_txt)
        self.assertIn("Neighbours Evaluated       234", stats_txt)
        self.assertIn("Converged?                 No", stats_txt)
        self.assertIn("Estimated Search Space", stats_txt)
        self.assertIn("≈ 4.0^57", stats_txt)
        self.assertIn("Research History", stats_txt)
        self.assertIn("Exp-025", stats_txt)
        hist = format_research_history(stats["research_history"])
        self.assertIn("Round", hist)
        self.assertIn("Exp-150", hist)

        live = format_dashboard(
            {
                "round": 2,
                "max_rounds": 8,
                "strategy_label": "Hill Climbing",
                "current_baseline_id": "Exp-150",
                "current_score": 0.99272,
                "neighbours_evaluated": 234,
                "best_improvement": 0.00081,
                "last_improvement": 0.00012,
                "converged": False,
                "families_tested": 56,
                "n_families_total": 57,
                "search_space": {"n_families": 57, "avg_candidates": 4.2, "possible_txt": "≈ 4.2^57"},
                "exploration_rate_txt": "~0%",
            }
        )
        self.assertIn("Search State", live)
        self.assertIn("Current Champion", live)
        self.assertIn("Exp-150", live)
        self.assertIn("Converged?               No", live)


if __name__ == "__main__":
    unittest.main()
