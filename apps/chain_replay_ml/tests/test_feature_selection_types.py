"""Tests for Phase 4B.0: Feature Selection Types, Contracts, Enums, and Configuration."""

from __future__ import annotations

import unittest

from chain_replay_ml.feature_selection.types import (
    AttributionStage,
    CanonicalFeatureAction,
    CompositeAttributionResult,
    CompositeProvenanceRecord,
    CompositeSelectionConfig,
    CompositeStrategy,
    DEFAULT_COMPOSITE_SELECTION_CONFIG,
    DiscoveryDiagnosticAction,
    FeatureAttributionRecord,
    ValidationDiagnosticAction,
    map_discovery_action_to_canonical,
    map_validation_action_to_canonical,
)


class TestFeatureSelectionTypes(unittest.TestCase):
    def test_canonical_feature_action_enum(self) -> None:
        """Verify canonical macro-lifecycle governance enum has exactly KEEP, WATCH, REMOVE."""
        self.assertEqual(set(a.value for a in CanonicalFeatureAction), {"KEEP", "WATCH", "REMOVE"})
        self.assertEqual(CanonicalFeatureAction.KEEP, "KEEP")
        self.assertEqual(CanonicalFeatureAction.WATCH, "WATCH")
        self.assertEqual(CanonicalFeatureAction.REMOVE, "REMOVE")

    def test_diagnostic_action_enums(self) -> None:
        """Verify Stage-1 and Stage-2 diagnostic enums."""
        self.assertEqual(DiscoveryDiagnosticAction.KEEP, "KEEP")
        self.assertEqual(DiscoveryDiagnosticAction.REVIEW_FAMILY, "REVIEW FAMILY")
        self.assertEqual(DiscoveryDiagnosticAction.MERGE_CANDIDATE, "MERGE CANDIDATE")
        self.assertEqual(DiscoveryDiagnosticAction.RETIRE_CANDIDATE, "RETIRE CANDIDATE")

        self.assertEqual(ValidationDiagnosticAction.PRODUCTION_READY, "PRODUCTION READY")
        self.assertEqual(ValidationDiagnosticAction.NEEDS_REVIEW, "NEEDS REVIEW")
        self.assertEqual(ValidationDiagnosticAction.UNSTABLE, "UNSTABLE")

        self.assertEqual(AttributionStage.STAGE_DISCOVERY, "discovery")
        self.assertEqual(AttributionStage.STAGE_VALIDATION, "validation")
        self.assertEqual(CompositeStrategy.COMPOSITE_NONLINEAR, "composite_nonlinear")

    def test_discovery_diagnostic_to_canonical_mapping(self) -> None:
        """Verify exact mapping from Stage-1 diagnostic actions to canonical lifecycle."""
        self.assertEqual(map_discovery_action_to_canonical(DiscoveryDiagnosticAction.KEEP), CanonicalFeatureAction.KEEP)
        self.assertEqual(map_discovery_action_to_canonical(DiscoveryDiagnosticAction.REVIEW_FAMILY), CanonicalFeatureAction.WATCH)
        self.assertEqual(map_discovery_action_to_canonical(DiscoveryDiagnosticAction.MERGE_CANDIDATE), CanonicalFeatureAction.WATCH)
        self.assertEqual(map_discovery_action_to_canonical(DiscoveryDiagnosticAction.RETIRE_CANDIDATE), CanonicalFeatureAction.REMOVE)

        # Test string aliases
        self.assertEqual(map_discovery_action_to_canonical("KEEP"), CanonicalFeatureAction.KEEP)
        self.assertEqual(map_discovery_action_to_canonical("REVIEW"), CanonicalFeatureAction.WATCH)
        self.assertEqual(map_discovery_action_to_canonical("REVIEW FAMILY"), CanonicalFeatureAction.WATCH)
        self.assertEqual(map_discovery_action_to_canonical("FAMILY DECISION REQUIRED"), CanonicalFeatureAction.WATCH)
        self.assertEqual(map_discovery_action_to_canonical("MERGE"), CanonicalFeatureAction.WATCH)
        self.assertEqual(map_discovery_action_to_canonical("MERGE CANDIDATE"), CanonicalFeatureAction.WATCH)
        self.assertEqual(map_discovery_action_to_canonical("RETIRE"), CanonicalFeatureAction.REMOVE)
        self.assertEqual(map_discovery_action_to_canonical("RETIRE CANDIDATE"), CanonicalFeatureAction.REMOVE)
        self.assertEqual(map_discovery_action_to_canonical("UNKNOWN_ACTION"), CanonicalFeatureAction.WATCH)

    def test_validation_diagnostic_to_canonical_mapping(self) -> None:
        """Verify exact mapping from Stage-2 validation diagnostic actions to canonical lifecycle."""
        self.assertEqual(map_validation_action_to_canonical(ValidationDiagnosticAction.PRODUCTION_READY), CanonicalFeatureAction.KEEP)
        self.assertEqual(map_validation_action_to_canonical(ValidationDiagnosticAction.NEEDS_REVIEW), CanonicalFeatureAction.WATCH)
        self.assertEqual(map_validation_action_to_canonical(ValidationDiagnosticAction.UNSTABLE), CanonicalFeatureAction.REMOVE)

        # Test string aliases
        self.assertEqual(map_validation_action_to_canonical("PRODUCTION READY"), CanonicalFeatureAction.KEEP)
        self.assertEqual(map_validation_action_to_canonical("NEEDS REVIEW"), CanonicalFeatureAction.WATCH)
        self.assertEqual(map_validation_action_to_canonical("UNSTABLE"), CanonicalFeatureAction.REMOVE)
        self.assertEqual(map_validation_action_to_canonical("UNKNOWN"), CanonicalFeatureAction.WATCH)

    def test_composite_selection_config_defaults(self) -> None:
        """Verify explicit defaults of CompositeSelectionConfig."""
        cfg = DEFAULT_COMPOSITE_SELECTION_CONFIG
        self.assertEqual(cfg.weight_mi_pre, 0.45)
        self.assertEqual(cfg.weight_perm_pre, 0.40)
        self.assertEqual(cfg.weight_shap_post, 0.45)
        self.assertEqual(cfg.weight_perm_post, 0.30)
        self.assertEqual(cfg.weight_mi_post, 0.25)
        self.assertEqual(cfg.corr_threshold, 0.95)
        self.assertEqual(cfg.extreme_duplicate_threshold, 0.999)
        self.assertEqual(cfg.moderate_corr_threshold, 0.85)
        self.assertEqual(cfg.max_null_pct, 5.0)
        self.assertEqual(cfg.min_coverage_pct, 90.0)
        self.assertEqual(cfg.high_band_pct, 66.67)
        self.assertEqual(cfg.low_band_pct, 33.33)
        self.assertEqual(cfg.max_subsample_rows, 10_000)
        self.assertEqual(cfg.random_seed, 42)

    def test_composite_selection_config_validation(self) -> None:
        """Verify validation gates in CompositeSelectionConfig."""
        with self.assertRaises(ValueError):
            CompositeSelectionConfig(weight_mi_pre=0.0, weight_perm_pre=0.0)

        with self.assertRaises(ValueError):
            CompositeSelectionConfig(weight_shap_post=0.0, weight_perm_post=0.0, weight_mi_post=0.0)

        with self.assertRaises(ValueError):
            CompositeSelectionConfig(corr_threshold=1.5)

        with self.assertRaises(ValueError):
            CompositeSelectionConfig(max_null_pct=-1.0)

        with self.assertRaises(ValueError):
            CompositeSelectionConfig(max_subsample_rows=0)

    def test_attribution_records_instantiation(self) -> None:
        """Verify FeatureAttributionRecord and CompositeAttributionResult instantiation."""
        record = FeatureAttributionRecord(
            feature_name="svi_param_b",
            stage=AttributionStage.STAGE_DISCOVERY,
            mi_raw=0.082,
            perm_importance_raw=0.015,
            mi_pct=85.0,
            perm_pct=78.0,
            composite_score=81.8,
            composite_rank=1,
            diagnostic_action=DiscoveryDiagnosticAction.KEEP.value,
            canonical_action=CanonicalFeatureAction.KEEP,
            confidence="High",
            reason="High MI and Permutation",
        )
        self.assertEqual(record.feature_name, "svi_param_b")
        self.assertEqual(record.canonical_action, CanonicalFeatureAction.KEEP)
        self.assertEqual(record.stage, AttributionStage.STAGE_DISCOVERY)

        result = CompositeAttributionResult(
            run_id="test_run_01",
            dataset_id="test_ds_01",
            target_column="future_ltp_1m",
            stage=AttributionStage.STAGE_DISCOVERY,
            strategy=CompositeStrategy.COMPOSITE_NONLINEAR,
            total_features_evaluated=10,
            selected_feature_count=3,
            selected_features=["svi_param_b", "sabr_param_nu", "ultima"],
            quarantined_features=["svi_param_a"],
            pruned_collinear_features=["zomma"],
            attributions={"svi_param_b": record},
        )
        self.assertEqual(result.total_features_evaluated, 10)
        self.assertEqual(result.selected_feature_count, 3)
        self.assertEqual(len(result.selected_features), 3)

    def test_provenance_record_instantiation(self) -> None:
        """Verify CompositeProvenanceRecord instantiation."""
        prov = CompositeProvenanceRecord(
            run_id="test_run_01",
            dataset_id="test_ds_01",
            strategy="composite_nonlinear",
            config_json='{"corr_threshold": 0.95}',
            sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            input_feature_count=10,
            selected_feature_count=3,
            selected_features=["svi_param_b", "sabr_param_nu", "ultima"],
            created_at_iso="2026-08-22T19:50:00Z",
        )
        self.assertEqual(prov.run_id, "test_run_01")
        self.assertEqual(prov.selected_feature_count, 3)


if __name__ == "__main__":
    unittest.main()
