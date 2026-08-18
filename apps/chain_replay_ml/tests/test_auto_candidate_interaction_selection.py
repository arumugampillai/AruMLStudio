"""Tests for Auto Candidate Generation Interaction-Parent Selection & Commutative Deduplication."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unittest
from typing import Any

from chain_replay_ml.dataset_builder.feature_domains import primary_domain_of
from chain_replay_ml.dataset_builder.feature_registry_store import (
    save_store as save_feature_store,
)
from chain_replay_ml.dataset_builder.transformations.interaction_ui import (
    bulk_interaction_pairs,
    is_commutative_interaction_op,
)
from chain_replay_ml.production_validation.api import (
    DatasetContext,
    build_dataset_context,
    rank_features_for_candidate_generation,
)
from chain_replay_ml.production_validation.evidence_store import (
    append_validation_evidence,
    get_connection,
)
from master_dataset_tk.auto_candidate_generation import (
    build_auto_candidate_transformation_config,
    default_candidate_generation_prefs,
    select_interaction_parent_features,
)


class TestAutoCandidateInteractionSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_candidate_sel_test_")
        self.chart_dir = os.path.join(self.tmp_dir, "chart")
        self.data_dir = os.path.join(self.chart_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.ctx_nifty_3s = build_dataset_context(
            market="NIFTY", sampling_interval_sec=3, sliding_window="standard", feature_project_id="all"
        )
        self.ctx_sensex_1s = build_dataset_context(
            market="SENSEX", sampling_interval_sec=1, sliding_window="standard", feature_project_id="all"
        )

        # Baseline Feature Store with a deprecated feature
        feat_store = {
            "registry_version": "1.0",
            "feature_ids": {
                "opt_bid_ask_spread_abs": "FR0001",
                "volatility_depr": "FR0002",
            },
            "feature_identities": {
                "FR0001": {
                    "feature_id": "FR0001",
                    "name": "opt_bid_ask_spread_abs",
                    "domain": "price_premium",
                    "implementation_status": "implemented",
                },
                "FR0002": {
                    "feature_id": "FR0002",
                    "name": "volatility_depr",
                    "domain": "implied_volatility",
                    "implementation_status": "deprecated",
                },
            },
            "disabled_features": {},
            "deleted_feature_ids": {},
        }
        save_feature_store(self.data_dir, feat_store)

        # Evidence in NIFTY 3s for specific features
        conn = get_connection(self.data_dir)
        try:
            models = ["model_xgb", "model_lgbm", "model_nn", "model_catboost"]
            # 1. High-performing IV feature (Promotion Candidate in NIFTY 3s)
            iv_rows = [
                {
                    "feature_name": "current_iv",
                    "feature_source": "registry",
                    "pipeline_id": "PL_0001",
                    "pipeline_snapshot_id": "SNP_1",
                    "model_name": models[i % 4],
                    "recommendation": "KEEP",
                    "permutation_mean": 0.090,
                    "importance_rank": 1,
                    "run_timestamp": f"2026-08-18T1{i:02d}:00:00Z",
                }
                for i in range(8)
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_3s, evidence_rows=iv_rows)

            # 2. High-performing Greeks feature (Train Candidate in NIFTY 3s)
            greek_rows = [
                {
                    "feature_name": "delta",
                    "feature_source": "registry",
                    "pipeline_id": "PL_0001",
                    "pipeline_snapshot_id": "SNP_1",
                    "model_name": models[i % 2],
                    "recommendation": "KEEP",
                    "permutation_mean": 0.075,
                    "importance_rank": 2,
                    "run_timestamp": f"2026-08-18T1{i:02d}:00:00Z",
                }
                for i in range(6)
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_3s, evidence_rows=greek_rows)
        finally:
            conn.close()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_interaction_parent_selection_not_alphabetical(self) -> None:
        """Verify features from diverse domains (such as IV, Greeks, Open Interest) are selected, not just 'a'...'d'."""
        # Create a large pool of canonical features across domains
        pool = [
            "option_bid", "option_ask", "option_open", "option_high", "option_low", "option_vwap",  # price_premium (starts with o)
            "spot_price", "spot_return", "spot_vwap", "spot_high", "spot_low", "spot_open",         # spot_futures (starts with s)
            "delta", "gamma", "vega", "theta", "rho",                                                # greeks (starts with d, g, v, t, r)
            "current_iv", "iv_slope", "iv_skew", "iv_atm", "iv_otm",                                 # implied_volatility (starts with c, i)
            "total_oi", "oi_change", "pe_oi", "ce_oi", "pcr_oi",                                     # open_interest (starts with t, o, p, c)
            "total_volume", "volume_flow", "buy_volume", "sell_volume", "vwap_distance",             # volume_liquidity (starts with t, v, b, s)
        ]
        selected = select_interaction_parent_features(
            features=pool,
            data_dir=self.data_dir,
            context=self.ctx_nifty_3s,
            max_parents=18,
        )
        self.assertEqual(len(selected), 18)

        # Check domains represented in selected
        domains_selected = {primary_domain_of(f) if f in ("delta", "current_iv", "option_bid", "option_ask", "option_vwap") else f.split("_")[0] for f in selected}
        self.assertGreaterEqual(len(domains_selected), 4)

    def test_02_commutative_interaction_deduplication(self) -> None:
        """Verify commutative operators (multiply, add, absdiff) produce only A <= B canonical pairs."""
        feats = ["delta", "current_iv", "spot_price"]

        # Multiply
        mul_pairs = bulk_interaction_pairs(feats, feats, op="multiply", skip_identical=True, symmetric_canonical=True)
        mul_outputs = [p["output"] for p in mul_pairs]
        self.assertEqual(len(mul_outputs), 3)  # 3 * 2 // 2 = 3 pairs
        self.assertIn("current_iv_x_delta", mul_outputs)  # "current_iv" < "delta"
        self.assertNotIn("delta_x_current_iv", mul_outputs)

        # Add
        add_pairs = bulk_interaction_pairs(feats, feats, op="add", skip_identical=True, symmetric_canonical=True)
        add_outputs = [p["output"] for p in add_pairs]
        self.assertEqual(len(add_outputs), 3)
        self.assertIn("current_iv_plus_delta", add_outputs)
        self.assertNotIn("delta_plus_current_iv", add_outputs)

        # Absolute difference
        absdiff_pairs = bulk_interaction_pairs(feats, feats, op="absolute_difference", skip_identical=True, symmetric_canonical=True)
        absdiff_outputs = [p["output"] for p in absdiff_pairs]
        self.assertEqual(len(absdiff_outputs), 3)
        self.assertIn("current_iv_absdiff_delta", absdiff_outputs)
        self.assertNotIn("delta_absdiff_current_iv", absdiff_outputs)

    def test_03_asymmetric_interaction_preserved(self) -> None:
        """Verify asymmetric operators (divide, subtract) preserve both directional pairs."""
        feats = ["delta", "current_iv"]

        # Divide
        div_pairs = bulk_interaction_pairs(feats, feats, op="divide", skip_identical=True, symmetric_canonical=True)
        div_outputs = [p["output"] for p in div_pairs]
        self.assertEqual(len(div_outputs), 2)
        self.assertIn("delta_div_current_iv", div_outputs)
        self.assertIn("current_iv_div_delta", div_outputs)

        # Subtract
        sub_pairs = bulk_interaction_pairs(feats, feats, op="subtract", skip_identical=True, symmetric_canonical=True)
        sub_outputs = [p["output"] for p in sub_pairs]
        self.assertEqual(len(sub_outputs), 2)
        self.assertIn("delta_minus_current_iv", sub_outputs)
        self.assertIn("current_iv_minus_delta", sub_outputs)

    def test_04_deprecated_features_excluded_from_interaction_parents(self) -> None:
        """Verify deprecated features are excluded from interaction parent selection."""
        feats = ["current_iv", "volatility_depr", "delta"]
        selected = select_interaction_parent_features(
            features=feats,
            data_dir=self.data_dir,
            context=self.ctx_nifty_3s,
            max_parents=10,
        )
        self.assertIn("current_iv", selected)
        self.assertIn("delta", selected)
        self.assertNotIn("volatility_depr", selected)

    def test_05_cold_start_parent_selection_fallback(self) -> None:
        """Verify cold-start context with zero evidence selects balanced domain distribution without error."""
        empty_dir = os.path.join(self.tmp_dir, "empty_data")
        os.makedirs(empty_dir, exist_ok=True)
        pool = [
            "option_bid", "option_ask", "option_vwap",
            "spot_price", "spot_return", "spot_vwap",
            "delta", "gamma", "vega",
            "current_iv", "iv_slope", "iv_skew",
        ]
        selected = select_interaction_parent_features(
            features=pool,
            data_dir=empty_dir,
            context=self.ctx_sensex_1s,
            max_parents=8,
        )
        self.assertEqual(len(selected), 8)
        domains = {f.split("_")[0] for f in selected}
        self.assertGreaterEqual(len(domains), 3)

    def test_06_context_isolation_in_parent_selection(self) -> None:
        """Verify evidence in NIFTY 3s does not leak into SENSEX 1s ranking."""
        ranked_nifty = rank_features_for_candidate_generation(
            self.data_dir,
            features=["current_iv", "total_volume"],
            context=self.ctx_nifty_3s,
        )
        ranked_sensex = rank_features_for_candidate_generation(
            self.data_dir,
            features=["current_iv", "total_volume"],
            context=self.ctx_sensex_1s,
        )

        # In NIFTY 3s, current_iv has strong evidence (PROMOTION / KEEP streak)
        nifty_decisions = {k: v.decision for k, v in ranked_nifty}
        self.assertIn(nifty_decisions["current_iv"], ["PROMOTION_CANDIDATE_QUALIFIED", "TRAIN_CANDIDATE"])

        # In SENSEX 1s, current_iv has zero runs -> NEW_UNSEEN
        sensex_decisions = {k: v.decision for k, v in ranked_sensex}
        self.assertEqual(sensex_decisions["current_iv"], "NEW_UNSEEN")

    def test_07_deterministic_ordering(self) -> None:
        """Verify identical candidate pool produces strictly deterministic parent selections."""
        pool = [
            "delta", "current_iv", "spot_price", "total_oi", "total_volume", "option_bid",
            "gamma", "iv_skew", "spot_vwap", "oi_change", "volume_flow", "option_vwap"
        ]
        run1 = select_interaction_parent_features(pool, self.data_dir, self.ctx_nifty_3s, max_parents=6)
        run2 = select_interaction_parent_features(pool, self.data_dir, self.ctx_nifty_3s, max_parents=6)
        self.assertEqual(run1, run2)

    def test_08_domain_redistribution_when_domain_has_fewer_than_quota(self) -> None:
        """Verify slots from small domains are redistributed deterministically to other active domains."""
        pool = [
            "delta",  # greeks has only 1 feature
            "current_iv", "iv_slope", "iv_skew", "iv_atm", "iv_otm",  # IV has 5 features
            "spot_price", "spot_return", "spot_vwap", "spot_high",   # spot has 4 features
        ]
        selected = select_interaction_parent_features(pool, self.data_dir, self.ctx_nifty_3s, max_parents=8)
        self.assertEqual(len(selected), 8)
        self.assertIn("delta", selected)

    def test_09_promotion_and_train_candidate_priority(self) -> None:
        """Verify promotion candidates and train candidates are placed first within their domain."""
        # current_iv is a promotion candidate, iv_slope is unseen
        pool = ["iv_slope", "current_iv", "iv_skew"]
        selected = select_interaction_parent_features(pool, self.data_dir, self.ctx_nifty_3s, max_parents=2)
        self.assertEqual(selected[0], "current_iv")

    def test_10_exclude_filtering(self) -> None:
        """Verify EXCLUDE features with is_candidate_generation_allowed=False are never selected."""
        ranked = rank_features_for_candidate_generation(
            self.data_dir,
            features=["volatility_depr"],
            context=self.ctx_nifty_3s,
        )
        self.assertEqual(len(ranked), 1)
        self.assertFalse(ranked[0][1].is_candidate_generation_allowed)

    def test_11_zero_evidence_feature_inclusion(self) -> None:
        """Verify NEW_UNSEEN feature is included when pool has capacity."""
        pool = ["total_volume", "delta"]
        selected = select_interaction_parent_features(pool, self.data_dir, self.ctx_nifty_3s, max_parents=5)
        self.assertIn("total_volume", selected)
        self.assertIn("delta", selected)

    def test_12_small_feature_pool_backward_compatibility(self) -> None:
        """Verify small pools (<= max_parents) return all allowed features intact."""
        small_pool = ["delta", "current_iv", "spot_price"]
        selected = select_interaction_parent_features(small_pool, self.data_dir, self.ctx_nifty_3s, max_parents=36)
        self.assertEqual(set(selected), set(small_pool))

    def test_13_auto_config_builder_uses_upgraded_selection(self) -> None:
        """Verify build_auto_candidate_transformation_config integrates parent selection and deduplication."""
        feats = ["delta", "current_iv", "spot_price", "total_volume"]
        prefs = default_candidate_generation_prefs()
        config = build_auto_candidate_transformation_config(
            features=feats,
            interval_sec=3,
            candidate_prefs=prefs,
            data_dir=self.data_dir,
        )
        transforms = config.get("transformations") or []
        ix_trans = [t for t in transforms if t.get("id") == "interaction"]
        self.assertEqual(len(ix_trans), 1)
        pairs = ix_trans[0].get("params", {}).get("pairs") or []
        outputs = [p["output"] for p in pairs]

        # Commutative multiply has canonical name only
        self.assertIn("current_iv_x_delta", outputs)
        self.assertNotIn("delta_x_current_iv", outputs)

        # Asymmetric divide has both directions
        self.assertIn("current_iv_div_delta", outputs)
        self.assertIn("delta_div_current_iv", outputs)


if __name__ == "__main__":
    unittest.main()

