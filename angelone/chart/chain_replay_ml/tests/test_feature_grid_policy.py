"""Tests for feature grid parity policy."""

from __future__ import annotations

from dataclasses import dataclass

from chain_replay_ml.dataset_builder.feature_grid_policy import (
    FeatureComputationKind,
    FeatureSharedScope,
    build_feature_parity_spec,
    classify_feature,
    feature_parity_audit_rows,
    feature_parity_for_name,
    resolve_feature_grid_step_sec,
    rv_subsample_step_sec,
    validate_warmup_units,
)
from chain_replay_ml.features_atm_band import get_realized_volatility


@dataclass
class _Ctx:
    feature_grid_step_sec: int


def test_resolve_feature_grid_step_from_ctx():
    assert resolve_feature_grid_step_sec(ctx=_Ctx(3)) == 3


def test_resolve_feature_grid_step_from_sampling():
    assert resolve_feature_grid_step_sec(sampling={"trainingIntervalSec": 15}) == 15


def test_explicit_ltp_ema20_row():
    row = feature_parity_for_name("ltp_ema20_to_ltp_ratio")
    assert row["kind"] == "grid_bar"
    assert row["warmup"] == "20 bars"
    assert "EMA20(ltp)" in row["depends_on"]
    assert "token" not in row["depends_on"]
    assert row["scope"] == "token"


def test_spot_scope_and_spot_ema_deps():
    spot = feature_parity_for_name("spot")
    assert spot["scope"] == "spot"
    row = feature_parity_for_name("spot_ema20_to_ltp_ratio")
    assert "EMA20(spot)" in row["depends_on"]
    assert "ltp" in row["depends_on"]
    assert "token" not in row["depends_on"]


def test_rv_features_have_explicit_rules():
    for name in ("opt_rv_5m", "opt_rv_10m", "opt_rv_ratio", "spot_rv_ratio", "spot_body_pct_prev1", "opt_body_pct_prev1"):
        row = feature_parity_for_name(name)
        assert "See feature implementation" not in row["rule"]
        assert row["rule"].strip()


def test_catalog_warmup_units_consistent():
    bad = []
    for row in feature_parity_audit_rows():
        spec = build_feature_parity_spec(str(row["feature"]))
        if not validate_warmup_units(spec):
            bad.append((row["feature"], spec.kind.value, spec.warmup))
    assert not bad, f"warmup unit mismatches: {bad[:5]}"


def test_no_see_implementation_in_catalog():
    rows = feature_parity_audit_rows()
    vague = [r["feature"] for r in rows if "See feature implementation" in str(r.get("rule") or "")]
    assert not vague, f"vague rules: {vague[:10]}"


def test_explicit_spot_change_5s_row():
    row = feature_parity_for_name("spot_change_5s")
    assert row["kind"] == "calendar_sec"
    assert row["warmup"] == "5 seconds"
    assert row["scope"] == "spot"


def test_straddle_zscore_is_grid_bar_not_sample_grid():
    row = feature_parity_for_name("atm_straddle_zscore_30m")
    assert row["kind"] == "grid_bar"
    assert "30" in row["warmup"]
    assert row["scope"] == "chain"


def test_chain_pcr_static_chain_scope():
    row = feature_parity_for_name("chain_pcr")
    assert row["kind"] == "static"
    assert row["scope"] == "chain"


def test_classify_static_greek():
    assert classify_feature("delta", "greeks") == FeatureComputationKind.STATIC


def test_inferred_calendar_warmup():
    spec = build_feature_parity_spec("ltp_return_30s", "price")
    assert spec.kind == FeatureComputationKind.CALENDAR_SEC
    assert spec.warmup == "30 seconds"


def test_rv_subsample_uses_grid_not_hardcoded_10():
    assert rv_subsample_step_sec(3) == 3.0
    step3 = rv_subsample_step_sec(3)
    step10 = rv_subsample_step_sec(10)
    assert int(300 / step3) > int(300 / step10)


def test_get_realized_volatility_accepts_grid_step():
    assert get_realized_volatility(None, 0.0, 300.0, grid_step_sec=3) is None


def test_scope_inference():
    assert build_feature_parity_spec("chain_pcr").scope == FeatureSharedScope.CHAIN
    assert build_feature_parity_spec("spot_change_1m", "momentum").scope == FeatureSharedScope.SPOT
