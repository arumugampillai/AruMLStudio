"""Tests for three-way feature pipeline parity helpers."""

from __future__ import annotations

from chain_replay_ml.dataset_builder.feature_pipeline_parity import (
    compare_three_paths,
    series_to_feature_dict,
    values_close,
)
import pandas as pd


def test_values_close_nulls():
    ok, diff = values_close("ltp", None, None)
    assert ok is True
    assert diff == 0.0


def test_values_close_numeric():
    ok, diff = values_close("delta", 0.45, 0.4500001)
    assert ok is True
    assert diff is not None


def test_series_to_feature_dict_skips_metadata():
    row = pd.Series({
        "token": "123",
        "timestamp": 1.0,
        "ltp": 100.5,
        "delta": 0.4,
        "_feature_raw": {"x": 1},
    })
    out = series_to_feature_dict(row)
    assert "token" not in out
    assert "timestamp" not in out
    assert out["delta"] == 0.4
    assert "_feature_raw" not in out


def test_compare_three_paths_all_match():
    feats = {"a": 1.0, "b": 2.0, "c": None}
    report = compare_three_paths(feats, feats, feats, ["a", "b", "c"])
    assert report["status"] == "pass"
    assert report["match_count"] == 3
    assert report["mismatch_count"] == 0


def test_compare_three_paths_detects_mismatch():
    d = {"a": 1.0, "b": 2.0}
    r = {"a": 1.0, "b": 2.01}
    l = {"a": 1.0, "b": 2.0}
    report = compare_three_paths(d, r, l, ["a", "b"])
    assert report["status"] == "fail"
    assert report["mismatch_count"] == 1
    assert report["mismatches"][0]["feature"] == "b"
