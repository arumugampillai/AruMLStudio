"""Tests for retrain dataset compatibility checks."""

from unittest.mock import patch

from chain_replay_ml.training.retrain_compatibility import (
    build_retrain_profile,
    build_retrain_profile_from_model,
    compare_dataset_to_retrain_profile,
    evaluate_retrain_dataset_choice,
    list_retrain_compatible_datasets,
)


def _base_meta(*, band=15, interval=3, market="NIFTY", targets=None):
    return {
        "strike_selection": {"mode": "atm_band", "band": band},
        "sampling": {"interval_sec": interval},
        "market": market,
        "prediction_type": "regression",
        "prediction_target_columns": targets or ["future_ltp_5m"],
    }


def test_compatible_dataset_passes_all_required_checks():
    profile = build_retrain_profile(
        target="future_ltp_5m",
        prediction_type="regression",
        dataset_meta=_base_meta(),
    )
    result = compare_dataset_to_retrain_profile(profile, _base_meta(), dataset_name="ds_a")
    assert result["compatible"] is True
    assert result["score_pct"] == 100
    assert all(c["passed"] for c in result["checks"] if c["required"])


def test_strike_band_mismatch_fails():
    profile = build_retrain_profile(
        target="future_ltp_5m",
        prediction_type="regression",
        dataset_meta=_base_meta(band=15),
    )
    result = compare_dataset_to_retrain_profile(
        profile,
        _base_meta(band=10),
        dataset_name="ds_b",
    )
    assert result["compatible"] is False
    strike = next(c for c in result["checks"] if c["id"] == "strike_selection")
    assert strike["passed"] is False


def test_missing_target_column_fails():
    profile = build_retrain_profile(
        target="future_ltp_5m",
        prediction_type="regression",
        dataset_meta=_base_meta(),
    )
    result = compare_dataset_to_retrain_profile(
        profile,
        _base_meta(targets=["future_ltp_10m"]),
        dataset_name="ds_c",
    )
    assert result["compatible"] is False
    target = next(c for c in result["checks"] if c["id"] == "target")
    assert target["passed"] is False


def test_pipeline_fingerprint_atm_band_overrides_stale_strike_selection():
    from chain_replay_ml.dataset_builder.expected_spec import resolve_atm_band, strike_selection_display_label

    meta = {
        "strike_selection": {"mode": "ATM_BAND", "band": 10},
        "pipeline_fingerprint": {"atm_band": 15},
    }
    assert resolve_atm_band(meta) == 15
    assert strike_selection_display_label(meta) == "ATM ±15"


def test_retrain_profile_when_source_dataset_deleted_uses_replay_config():
    detail = {
        "config": {
            "dataset": "MS_239f_3s_1340",
            "target": "future_ltp_5m",
            "prediction_type": "regression",
            "replay_config": {
                "market": "NIFTY",
                "sampling": {"interval_sec": 3},
                "strike_selection": {"mode": "atm_band", "band": 15},
                "prediction_target_columns": ["future_ltp_5m"],
            },
        },
        "pipeline_fingerprint": {"atm_band": 15, "sampling_interval_sec": 3},
    }
    with patch("chain_replay_ml.training.retrain_compatibility.load_model_detail", return_value=detail):
        with patch("chain_replay_ml.training.retrain_compatibility.load_dataset_metadata_json", return_value={}):
            profile = build_retrain_profile_from_model("/data", "my_model")
    assert profile["source_dataset_missing"] is True
    assert profile["target"] == "future_ltp_5m"
    assert profile["sampling_interval_sec"] == 3
    assert profile["strike_band"] == 15


def test_evaluate_missing_candidate_dataset_returns_incompatible_not_error():
    detail = {
        "config": {
            "dataset": "MS_old",
            "target": "future_ltp_5m",
            "prediction_type": "regression",
            "replay_config": {
                "sampling": {"interval_sec": 3},
                "strike_selection": {"mode": "atm_band", "band": 15},
                "prediction_target_columns": ["future_ltp_5m"],
            },
        },
    }

    def _meta(_data_dir: str, name: str) -> dict:
        return {} if name == "MS_missing" else _base_meta()

    with patch("chain_replay_ml.training.retrain_compatibility.load_model_detail", return_value=detail):
        with patch("chain_replay_ml.training.retrain_compatibility.load_dataset_metadata_json", side_effect=_meta):
            doc = evaluate_retrain_dataset_choice(
                "/data",
                source_model="my_model",
                dataset_name="MS_missing",
            )
    assert doc["compatibility"]["compatible"] is False
    assert doc["compatibility"]["checks"][0]["id"] == "dataset_available"


def test_list_compatible_skips_deleted_source_as_default():
    detail = {
        "config": {
            "dataset": "MS_deleted",
            "target": "future_ltp_5m",
            "prediction_type": "regression",
            "replay_config": {
                "market": "NIFTY",
                "sampling": {"interval_sec": 3},
                "strike_selection": {"mode": "atm_band", "band": 15},
                "prediction_target_columns": ["future_ltp_5m"],
            },
        },
    }
    registry_rows = [
        {
            "dataset_name": "MS_new",
            "has_parquet": True,
            "is_draft": False,
            "row_count": 1000,
            "day_count": 5,
            "market": "NIFTY",
        },
    ]

    def _meta(data_dir: str, name: str) -> dict:
        if name == "MS_deleted":
            return {}
        return _base_meta()

    with patch("chain_replay_ml.training.retrain_compatibility.load_model_detail", return_value=detail):
        with patch("chain_replay_ml.training.retrain_compatibility.load_dataset_metadata_json", side_effect=_meta):
            with patch("chain_replay_ml.dataset_builder.auditor.list_datasets", return_value=registry_rows):
                doc = list_retrain_compatible_datasets("/data", "my_model")
    assert doc["source_dataset_missing"] is True
    assert doc["default_dataset"] == "MS_new"
    assert doc["compatible_count"] == 1
