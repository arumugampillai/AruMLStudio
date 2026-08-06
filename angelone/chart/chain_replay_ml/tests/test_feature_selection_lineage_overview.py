"""Feature Selection lineage for Model Registry Overview."""

from __future__ import annotations

from chain_replay_ml.dataset_builder.analysis_feature_selection import (
    STRATEGY_CORR_PERM,
    STRATEGY_HCA,
    build_feature_selection_lineage,
    feature_selection_overview_rows,
    extract_feature_selection_lineage,
)
from chain_replay_ml.training.config import normalize_training_config


def test_overview_rows_hca_includes_policy_and_families() -> None:
    lineage = build_feature_selection_lineage(
        {
            "strategy": STRATEGY_HCA,
            "strategy_label": "HCA + Correlation + Permutation",
            "representative_policy": "top_2",
            "representative_policy_label": "Top 2",
            "correlation_threshold": 0.97,
            "permutation_threshold": 0.001,
            "n_input_features": 396,
            "n_after_correlation": 220,
            "n_families": 57,
            "n_features": 108,
            "hash": "f18a73e9abcdef",
            "features": ["a", "b"],
            "discovery_bundle_id": "DB_20260729_001",
        },
        discovery_bundle_id="DB_20260729_001",
    )
    rows = dict(feature_selection_overview_rows(lineage))
    assert rows["Source"] == "Analysis"
    assert rows["Selection Strategy"] == "HCA + Correlation + Permutation"
    assert rows["Representative Policy"] == "Top 2"
    assert rows["Correlation Threshold"] == "0.97"
    assert rows["Permutation Threshold"] == "0.001"
    assert rows["Original Features"] == 396
    assert rows["After Correlation"] == 220
    assert rows["HCA Families"] == 57
    assert rows["Selected Features"] == 108
    assert rows["Feature Set Hash"] == "f18a73e9..."
    assert rows["Discovery Bundle"] == "DB_20260729_001"


def test_overview_rows_corr_perm_omits_hca_fields() -> None:
    lineage = build_feature_selection_lineage(
        {
            "strategy": STRATEGY_CORR_PERM,
            "correlation_threshold": 0.97,
            "permutation_threshold": 0.001,
            "n_input_features": 396,
            "n_after_correlation": 220,
            "n_features": 169,
            "hash": "a91f4c72zzzz",
            "features": ["x"],
            "n_families": 99,  # must still be omitted for non-HCA
        }
    )
    labels = [label for label, _ in feature_selection_overview_rows(lineage)]
    assert "Representative Policy" not in labels
    assert "HCA Families" not in labels
    rows = dict(feature_selection_overview_rows(lineage))
    assert rows["Selection Strategy"] == "Correlation + Permutation"
    assert rows["Selected Features"] == 169
    assert rows["Feature Set Hash"] == "a91f4c72..."


def test_training_config_preserves_analysis_feature_selection() -> None:
    raw = {
        "dataset": "ds1",
        "target": "future_ltp_5m",
        "algorithm": "xgboost",
        "features": ["a", "b"],
        "analysis_feature_selection": {
            "source": "analysis",
            "strategy": STRATEGY_CORR_PERM,
            "strategy_label": "Correlation + Permutation",
            "n_selected_features": 2,
            "features": ["a", "b"],
        },
    }
    cfg = normalize_training_config(raw)
    assert cfg.analysis_feature_selection is not None
    assert cfg.analysis_feature_selection["strategy"] == STRATEGY_CORR_PERM
    doc = cfg.to_dict()
    assert doc["analysis_feature_selection"]["n_selected_features"] == 2
    extracted = extract_feature_selection_lineage({"config": doc})
    assert extracted is not None
    assert extracted["strategy"] == STRATEGY_CORR_PERM


def test_extract_lineage_no_recursion_when_config_missing() -> None:
    """Missing config must not recurse via a fresh empty dict each call."""
    import sys

    old = sys.getrecursionlimit()
    sys.setrecursionlimit(64)
    try:
        out = extract_feature_selection_lineage(
            {
                "model_name": "m1",
                "feature_importance": [{"feature": "ltp", "importance_pct": 10}],
            }
        )
        assert out is None
        # Cyclic config graph
        a: dict = {"x": 1}
        b: dict = {"config": a}
        a["config"] = b
        assert extract_feature_selection_lineage({"config": a}) is None
    finally:
        sys.setrecursionlimit(old)
