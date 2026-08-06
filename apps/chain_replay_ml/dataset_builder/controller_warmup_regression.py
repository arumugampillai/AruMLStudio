"""Canonical first-valid-sample expectations for controller-owned features."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .rolling_controllers import CONTROLLER_OWNED_READINESS_FEATURES

_DEFAULT_GRID_STEP_SEC = 3.0

_LAG_SUFFIX_SEC: dict[str, float] = {
    "5s": 5.0,
    "10s": 10.0,
    "15s": 15.0,
    "30s": 30.0,
    "1m": 60.0,
    "3m": 180.0,
    "5m": 300.0,
    "15m": 900.0,
}

_LTP_EMA_RATIO_RE = re.compile(r"^ltp_ema(\d+)_to_(ltp|spot)_ratio$")
_LTP_EMA_LEVEL_RE = re.compile(r"^ltp_ema(\d+)$")
_SPOT_EMA_RATIO_RE = re.compile(r"^spot_ema(\d+)_to_ltp_ratio$")
_SPOT_EMA_LEVEL_RE = re.compile(r"^spot_ema(\d+)$")
_LAG_FEATURE_RE = re.compile(r"^ltp_to_spot_ratio_lag_(.+)$")
_DGT_LAG_RE = re.compile(r"^dgt_reiv_pred_lag_(.+)$")
_DGT_CHANGE_RE = re.compile(r"^dgt_reiv_pred_change_(.+)$")
_DGT_ERR_LAG_RE = re.compile(r"^dgt_prediction_error_lag_(.+)$")
_DGT_ERR_CHANGE_RE = re.compile(r"^dgt_prediction_error_change_(.+)$")
_LTP_RETURN_RE = re.compile(r"^ltp_return_(.+)$")
_SPOT_CHANGE_RE = re.compile(r"^spot_change_(.+)$")
_LTP_CHANGE_RE = re.compile(r"^ltp_change_(.+)$")


@dataclass(frozen=True)
class ControllerWarmupSpec:
    """One row in the controller warmup verification table."""

    label: str
    feature_name: str
    controller_id: str
    expected_first_valid: int


@dataclass(frozen=True)
class WarmupRegressionSpec:
    """One row in the expanded warmup regression table (UI + replay validation)."""

    feature_name: str
    category: str
    source: str


# Permanent regression table — policy-aligned @ 3s feature grid.
CONTROLLER_WARMUP_SPEC: tuple[ControllerWarmupSpec, ...] = (
    ControllerWarmupSpec("EMA9", "ltp_ema9", "token.ltp.ema9", 9),
    ControllerWarmupSpec("EMA20", "ltp_ema20", "token.ltp.ema20", 20),
    ControllerWarmupSpec("EMA50", "ltp_ema50", "token.ltp.ema50", 50),
    ControllerWarmupSpec("EMA100", "ltp_ema100", "token.ltp.ema100", 100),
    ControllerWarmupSpec("EMA200", "ltp_ema200", "token.ltp.ema200", 200),
    ControllerWarmupSpec("STD20", "ltp_std20", "token.ltp.std20", 20),
    ControllerWarmupSpec("RV5m", "opt_rv_5m", "token.rv.5m", 31),
    ControllerWarmupSpec("RV10m", "opt_rv_10m", "token.rv.10m", 61),
    ControllerWarmupSpec("IV z-score 1m", "iv_zscore_1m", "token.iv_window.1m", 20),
    ControllerWarmupSpec("IV z-score 5m", "iv_zscore_5m", "token.iv_window.5m", 100),
    ControllerWarmupSpec("IV z-score 15m", "iv_zscore_15m", "token.iv_window.15m", 300),
    ControllerWarmupSpec("IV rank session", "iv_rank_session", "token.iv_window.session", 1),
    # iv_change_* / iv_pct_change_1m → Pipeline Owned (Difference/Return)
)

CONTROLLER_WARMUP_FEATURE_NAMES: frozenset[str] = frozenset(
    spec.feature_name for spec in CONTROLLER_WARMUP_SPEC
)


def _calendar_first_valid(sec: float, step_sec: float) -> int:
    step = max(float(step_sec), 1.0)
    return int(math.ceil(sec / step) + 1)


def _iv_zscore_first_valid(cal_sec: float, step_sec: float) -> int:
    step = max(float(step_sec), 1.0)
    return int(cal_sec / step)


def _rv_first_valid(period_samples: int) -> int:
    return int(period_samples) + 1


def _lag_suffix_sec(suffix: str) -> float | None:
    key = str(suffix or "").strip().lower()
    return _LAG_SUFFIX_SEC.get(key)


def expected_first_valid_for_feature(
    feature_name: str,
    step_sec: float = _DEFAULT_GRID_STEP_SEC,
) -> int | None:
    """Resolve expected first valid sample for NullUntilReady features @ feature grid."""
    name = str(feature_name or "").strip()
    if not name:
        return None
    step = max(float(step_sec), 1.0)

    fixed: dict[str, int] = {
        "roll_iv": 1,
        "iv_drift_from_roll": 1,
        "iv_rank_session": 1,
        "ltp_std20": 20,
        "ltp_std20_to_ltp_ratio": 20,
        "opt_rv_5m": _rv_first_valid(30),
        "opt_rv_10m": _rv_first_valid(60),
        "spot_rv_5m": _rv_first_valid(30),
        "spot_rv_10m": _rv_first_valid(60),
        "opt_rv_ratio": _rv_first_valid(60),
        "spot_rv_ratio": _rv_first_valid(60),
        "iv_zscore_1m": _iv_zscore_first_valid(60.0, step),
        "iv_zscore_5m": _iv_zscore_first_valid(300.0, step),
        "iv_zscore_15m": _iv_zscore_first_valid(900.0, step),
        "iv_zscore_30m": _iv_zscore_first_valid(1800.0, step),
        "iv_change_1m": _calendar_first_valid(60.0, step),
        "iv_change_5m": _calendar_first_valid(300.0, step),
        "iv_change_15m": _calendar_first_valid(900.0, step),
        "iv_pct_change_1m": _calendar_first_valid(60.0, step),
    }
    if name in fixed:
        return fixed[name]

    m = _LTP_EMA_LEVEL_RE.match(name)
    if m:
        return int(m.group(1))

    m = _LTP_EMA_RATIO_RE.match(name)
    if m:
        return int(m.group(1))

    m = _SPOT_EMA_LEVEL_RE.match(name)
    if m:
        return int(m.group(1))

    m = _SPOT_EMA_RATIO_RE.match(name)
    if m:
        return int(m.group(1))

    m = _LAG_FEATURE_RE.match(name)
    if m:
        lag_sec = _lag_suffix_sec(m.group(1))
        if lag_sec is not None:
            return _calendar_first_valid(lag_sec, step)

    for pattern in (_LTP_RETURN_RE, _SPOT_CHANGE_RE, _LTP_CHANGE_RE):
        m = pattern.match(name)
        if m:
            lag_sec = _lag_suffix_sec(m.group(1))
            if lag_sec is not None:
                return _calendar_first_valid(lag_sec, step)

    for pattern in (_DGT_LAG_RE, _DGT_CHANGE_RE, _DGT_ERR_LAG_RE, _DGT_ERR_CHANGE_RE):
        m = pattern.match(name)
        if m:
            lag_sec = _lag_suffix_sec(m.group(1))
            if lag_sec is not None:
                return _calendar_first_valid(lag_sec, step)

    return None


def _warmup_regression_rows() -> tuple[WarmupRegressionSpec, ...]:
  rows: list[WarmupRegressionSpec] = []

  def add(feature: str, category: str, source: str) -> None:
      rows.append(WarmupRegressionSpec(feature, category, source))

  for feat, src in (
      ("roll_iv", "token.roll"),
      ("iv_drift_from_roll", "token.roll"),
  ):
      add(feat, "Roll/IV", src)
  for feat, src in (
      ("iv_zscore_30m", "token.iv_window.30m"),
  ):
      add(feat, "Roll/IV", src)

  for feat, src in (
      ("spot_rv_5m", "spot.rv.5m"),
      ("spot_rv_10m", "spot.rv.10m"),
      ("spot_rv_ratio", "spot.rv.ratio"),
      ("opt_rv_5m", "token.rv.5m"),
      ("opt_rv_10m", "token.rv.10m"),
      ("opt_rv_ratio", "token.rv.ratio"),
  ):
      add(feat, "RV", src)

  for period in (9, 20, 50, 100, 200):
      add(f"ltp_ema{period}", "EMA levels", f"token.ltp.ema{period}")
  add("ltp_std20", "EMA levels", "token.ltp.std20")
  for period in (9, 20, 50, 100, 200):
      add(f"spot_ema{period}", "EMA levels", f"spot.ema{period}")

  for suffix in ("30s", "1m", "3m", "5m", "15m"):
      # Pipeline Owned — not Master columns; retained for readiness math if selected.
      add(f"ltp_to_spot_ratio_lag_{suffix}", "Lag", "lookback")

  for feat in (
      "ltp_return_15s", "ltp_return_30s", "ltp_return_1m",
      "spot_change_15s", "spot_change_30s", "spot_change_1m",
      "ltp_change_1m", "ltp_change_5m", "ltp_change_15m",
      "iv_change_1m", "iv_change_5m", "iv_change_15m", "iv_pct_change_1m",
  ):
      # Pipeline Owned lookbacks — retained for readiness math if selected.
      add(feat, "Lookback", "pipeline")

  for feat in (
      "dgt_reiv_pred_lag_30s", "dgt_reiv_pred_lag_1m",
      "dgt_reiv_pred_change_30s", "dgt_reiv_pred_change_1m",
      "dgt_prediction_error_lag_30s",
  ):
      add(feat, "DGT", "pipeline")

  for spec in CONTROLLER_WARMUP_SPEC:
      if any(r.feature_name == spec.feature_name for r in rows):
          continue
      category = "Controller"
      if spec.feature_name.startswith("iv_zscore"):
          category = "Roll/IV"
      elif spec.feature_name.startswith(("opt_rv", "spot_rv")):
          category = "RV"
      elif "ema" in spec.feature_name or "std20" in spec.feature_name:
          category = "EMA levels"
      rows.append(WarmupRegressionSpec(spec.feature_name, category, spec.controller_id))

  return tuple(rows)


WARMUP_REGRESSION_SPEC: tuple[WarmupRegressionSpec, ...] = _warmup_regression_rows()

WARMUP_REGRESSION_FEATURE_NAMES: frozenset[str] = frozenset(
    spec.feature_name for spec in WARMUP_REGRESSION_SPEC
)

_WARMUP_REGRESSION_BY_FEATURE: dict[str, WarmupRegressionSpec] = {
    spec.feature_name: spec for spec in WARMUP_REGRESSION_SPEC
}


def is_valid_feature_value(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, str):
        s = val.strip().upper()
        if not s or s == "NULL":
            return False
        try:
            f = float(s)
        except ValueError:
            return False
        return not math.isnan(f)
    if isinstance(val, (int, float)):
        return not (isinstance(val, float) and math.isnan(val))
    return False


def first_valid_sample_from_series(
    samples: Sequence[tuple[int | None, Any]],
) -> int | None:
    """Return 1-based sample index of first non-NULL value."""
    for idx, (sample_no, val) in enumerate(samples):
        if is_valid_feature_value(val):
            if sample_no is not None:
                return int(sample_no)
            return idx + 1
    return None


def extract_feature_series_from_result(
    result: Any,
    feature_name: str,
) -> list[tuple[int | None, Any]]:
    """Pull per-trace-sample values for one feature from a simulation result."""
    from chain_replay_ml.feature_policy.warmup_calc_debug import (
        lookup_replay_values,
        resolve_replay_lookup_from_result,
    )

    trace = list(getattr(result, "full_trace", None) or [])
    lookup = resolve_replay_lookup_from_result(result)
    step_sec = int(max(getattr(result, "sampling_interval_sec", 3) or 3, 1))
    out: list[tuple[int | None, Any]] = []
    for tr in trace:
        ts = tr.get("ts")
        sample_no = tr.get("samples")
        try:
            sample_i = int(sample_no) if sample_no is not None else None
        except (TypeError, ValueError):
            sample_i = None
        val = None
        if lookup and ts is not None:
            replay_vals = lookup_replay_values(lookup, float(ts), step_sec=step_sec)
            val = replay_vals.get(feature_name) if replay_vals else None
        if val is None and feature_name == getattr(result, "feature_name", None):
            disp = tr.get("output_display")
            if disp not in (None, "NULL", "VALUE"):
                val = disp
        out.append((sample_i, val))
    return out


def validate_warmup_row_from_result(
    result: Any | None,
    spec: ControllerWarmupSpec,
) -> dict[str, Any]:
    """Validate one controller row against post-simulation replay data."""
    row: dict[str, Any] = {
        "label": spec.label,
        "feature": spec.feature_name,
        "controller": spec.controller_id,
        "expected": spec.expected_first_valid,
        "actual": None,
        "status": "PENDING",
        "note": "",
    }
    if result is None or not getattr(result, "ok", False):
        row["note"] = "Run simulation with Dataset maturity replay."
        return row

    lookup = (
        getattr(result, "all_features_lookup", None)
        or getattr(result, "maturity_replay_lookup", None)
    )
    if not lookup:
        row["status"] = "SKIP"
        row["note"] = "Enable Dataset maturity replay (or Calculate all features)."
        return row

    series = extract_feature_series_from_result(result, spec.feature_name)
    if not series:
        row["status"] = "SKIP"
        row["note"] = "No trace samples."
        return row

    actual = first_valid_sample_from_series(series)
    row["actual"] = actual
    trace_len = len(series)
    if actual is None:
        row["status"] = "FAIL"
        row["note"] = f"No valid value in {trace_len} sample(s); need ≥{spec.expected_first_valid}."
        return row

    for sample_no, val in series:
        if sample_no is None:
            continue
        if int(sample_no) < spec.expected_first_valid and is_valid_feature_value(val):
            row["status"] = "FAIL"
            row["note"] = f"Premature value at sample {sample_no}."
            return row

    if actual == spec.expected_first_valid:
        row["status"] = "PASS"
        return row

    row["status"] = "FAIL"
    row["note"] = f"Expected sample {spec.expected_first_valid}, got {actual}."
    return row


def validate_all_controller_warmups_from_result(
    result: Any | None,
) -> list[dict[str, Any]]:
    return [validate_warmup_row_from_result(result, spec) for spec in CONTROLLER_WARMUP_SPEC]


def validate_warmup_regression_row_from_result(
    result: Any | None,
    spec: WarmupRegressionSpec,
) -> dict[str, Any]:
    """Validate one expanded warmup row against post-simulation replay data."""
    step_sec = int(max(getattr(result, "sampling_interval_sec", _DEFAULT_GRID_STEP_SEC) or _DEFAULT_GRID_STEP_SEC, 1))
    expected = expected_first_valid_for_feature(spec.feature_name, step_sec=float(step_sec))
    row: dict[str, Any] = {
        "feature": spec.feature_name,
        "category": spec.category,
        "source": spec.source,
        "expected": expected,
        "actual": None,
        "status": "PENDING",
        "note": "",
    }
    if expected is None:
        row["status"] = "SKIP"
        row["note"] = "No warmup expectation for feature."
        return row
    if result is None or not getattr(result, "ok", False):
        row["note"] = "Run simulation with Dataset maturity replay."
        return row

    from chain_replay_ml.feature_policy.warmup_calc_debug import resolve_replay_lookup_from_result

    lookup = resolve_replay_lookup_from_result(result)
    if not lookup:
        row["status"] = "SKIP"
        replay_err = getattr(result, "maturity_replay_error", None)
        if replay_err and str(replay_err).strip():
            row["note"] = f"Replay unavailable: {replay_err}"
        else:
            row["note"] = "Enable Dataset maturity replay (or Calculate all features)."
        return row

    series = extract_feature_series_from_result(result, spec.feature_name)
    if not series:
        row["status"] = "SKIP"
        row["note"] = "No trace samples."
        return row

    actual = first_valid_sample_from_series(series)
    row["actual"] = actual
    trace_len = len(series)
    if actual is None:
        row["status"] = "FAIL"
        row["note"] = f"No valid value in {trace_len} sample(s); need ≥{expected}."
        return row

    for sample_no, val in series:
        if sample_no is None:
            continue
        if int(sample_no) < expected and is_valid_feature_value(val):
            row["status"] = "FAIL"
            row["note"] = f"Premature value at sample {sample_no}."
            return row

    if actual == expected:
        row["status"] = "PASS"
        return row

    row["status"] = "FAIL"
    row["note"] = f"Expected sample {expected}, got {actual}."
    return row


def validate_all_warmup_regressions_from_result(
    result: Any | None,
) -> list[dict[str, Any]]:
    return [
        validate_warmup_regression_row_from_result(result, spec)
        for spec in WARMUP_REGRESSION_SPEC
    ]


def warmup_regression_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "PENDING": 0}
    for row in rows:
        status = str(row.get("status") or "PENDING").upper()
        counts[status] = counts.get(status, 0) + 1
    return counts


def assert_spec_features_are_controller_owned() -> None:
    missing = CONTROLLER_WARMUP_FEATURE_NAMES - CONTROLLER_OWNED_READINESS_FEATURES
    if missing:
        raise AssertionError(
            f"CONTROLLER_WARMUP_SPEC features not in CONTROLLER_OWNED_READINESS_FEATURES: {sorted(missing)}",
        )


def measure_first_valid_from_builder(
    feature_name: str,
    *,
    max_samples: int = 400,
    step_sec: float = 3.0,
) -> int | None:
    """Synthetic builder path — used by permanent regression tests."""
    from chain_replay_ml.dataset_builder.chain_maps import ChainMaps
    from chain_replay_ml.dataset_builder.extended_features import (
        OptionFeatureState,
        enrich_with_chain_maps,
    )
    from chain_replay_ml.dataset_builder.rolling_controllers import (
        SpotControllers,
        update_token_iv_controllers,
        update_token_ltp_controllers,
        update_token_rv_controllers,
    )
    from chain_replay_ml.ticks import TickTimeline

    opt_state = OptionFeatureState()
    spot_ctrl = SpotControllers()
    open_ts = 1_700_000_000.0
    active = frozenset({feature_name, *CONTROLLER_WARMUP_FEATURE_NAMES})

    for i in range(max_samples):
        ts = open_ts + i * step_sec
        ltp = 100.0 + i * 0.1 + (i % 3) * 0.05
        spot = 25000.0 + i * 1.8 + (i % 5) * 0.5
        iv = 0.18 + (i % 7) * 0.002
        update_token_ltp_controllers(opt_state.controllers, ltp, ts=ts)
        update_token_rv_controllers(opt_state.controllers, ltp, ts=ts)
        update_token_iv_controllers(opt_state.controllers, iv, ts=ts)
        spot_ctrl.update(spot, ts=ts)
        enriched = enrich_with_chain_maps(
            {"ltp": ltp, "spot": spot},
            ts=ts,
            chain_maps=ChainMaps(),
            strike_mapping={},
            index_tl=TickTimeline(),
            atm_strike=25000,
            expiry_ts=open_ts + 86400.0,
            opt_state=opt_state,
            option_timeline=TickTimeline(),
            open_ts=open_ts,
            close_ts=open_ts + 3600.0 * 6,
            active_features=active,
            feature_grid_step_sec=step_sec,
            spot_controllers=spot_ctrl,
        )
        val = enriched.get(feature_name)
        if is_valid_feature_value(val):
            return i + 1
    return None


def run_permanent_warmup_regression() -> list[tuple[str, int, int | None, str]]:
    """Run all CONTROLLER_WARMUP_SPEC rows; return (label, expected, actual, status)."""
    assert_spec_features_are_controller_owned()
    max_n = max(spec.expected_first_valid for spec in CONTROLLER_WARMUP_SPEC) + 10
    out: list[tuple[str, int, int | None, str]] = []
    for spec in CONTROLLER_WARMUP_SPEC:
        actual = measure_first_valid_from_builder(
            spec.feature_name,
            max_samples=max_n,
        )
        if actual == spec.expected_first_valid:
            status = "PASS"
        else:
            status = "FAIL"
        out.append((spec.label, spec.expected_first_valid, actual, status))
    return out
