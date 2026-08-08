"""Build transformation_config that regenerates Pipeline Features (Phase 1A).

Master no longer emits the 212 pipeline-owned columns. Analysis Dataset builds
must recreate them via the Transformation Pipeline using Master-compatible names.
"""

from __future__ import annotations

from typing import Any, Sequence

from .feature_migration import get_migration_family, list_migration_families
from .transformations.config import normalize_transformation_config
from .transformations.interaction_ui import merge_interaction_into_config

_PARTS = ["trading_day", "token"]


def _hz(seconds: float, *, suffix: str | None = None, column: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"seconds": float(seconds)}
    if suffix is not None:
        out["suffix"] = str(suffix)
    if column is not None:
        out["column"] = str(column)
    return out


def _pair(
    left: str,
    right: str,
    op: str,
    output: str,
    *,
    scale: float = 1.0,
    eps: float | None = None,
) -> dict[str, Any]:
    p: dict[str, Any] = {
        "left": left,
        "right": right,
        "op": op,
        "output": output,
        "scale": float(scale),
    }
    if eps is not None:
        p["eps"] = float(eps)
    return p


def _lag_stage(feature: str, horizons: list[tuple[str, float]], interval: float) -> dict[str, Any]:
    return {
        "id": "lag",
        "enabled": True,
        "order": 10,
        "params": {
            "features": [feature],
            "horizons": [_hz(sec, suffix=suf) for suf, sec in horizons],
            "partition_by": list(_PARTS),
            "sample_interval_sec": float(interval),
        },
    }


def _diff_stage(
    features: list[str],
    horizons: list[dict[str, Any]],
    interval: float,
) -> dict[str, Any]:
    return {
        "id": "difference",
        "enabled": True,
        "order": 20,
        "params": {
            "features": features,
            "horizons": horizons,
            "partition_by": list(_PARTS),
            "sample_interval_sec": float(interval),
        },
    }


def _return_stage(
    features: list[str],
    horizons: list[dict[str, Any]],
    interval: float,
    *,
    scale: float = 100.0,
    denom_eps: float | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "features": features,
        "horizons": horizons,
        "scale": float(scale),
        "partition_by": list(_PARTS),
        "sample_interval_sec": float(interval),
    }
    if denom_eps is not None:
        params["denom_eps"] = float(denom_eps)
    return {
        "id": "return",
        "enabled": True,
        "order": 30,
        "params": params,
    }


def _time_shift_stages(interval: float) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []

    # Pure lags
    for fid in ("ltp_to_spot_ratio", "dgt_reiv_pred_lag", "dgt_prediction_error_lag"):
        fam = get_migration_family(fid)
        stages.append(_lag_stage(fam.base_feature, list(fam.horizons), interval))

    # Simple differences with one base + matching feature names
    for fid in (
        "ltp_to_spot_ratio_change",
        "dgt_reiv_pred_change",
        "ltp_change",
    ):
        fam = get_migration_family(fid)
        if fid == "ltp_to_spot_ratio_change":
            horizons = [
                _hz(sec, column=f"ltp_to_spot_ratio_change_{suf}")
                for suf, sec in fam.horizons
            ]
        elif fid == "dgt_reiv_pred_change":
            horizons = [
                _hz(sec, column=f"dgt_reiv_pred_change_{suf}")
                for suf, sec in fam.horizons
            ]
        else:
            horizons = [
                _hz(sec, column=f"ltp_change_{suf}")
                for suf, sec in fam.horizons
            ]
        stages.append(_diff_stage([fam.base_feature], horizons, interval))

    # Greeks @ 5m (one feature per stage — column override is not per-feature)
    for feat in ("delta", "gamma", "theta"):
        stages.append(_diff_stage(
            [feat],
            [_hz(300.0, column=f"{feat}_change_5m")],
            interval,
        ))
    # PCR
    stages.append(_diff_stage(
        ["chain_pcr"],
        [_hz(300.0, column="chain_pcr_change_5m")],
        interval,
    ))
    stages.append(_diff_stage(
        ["atm_pcr"],
        [_hz(300.0, column="atm_pcr_change_5m")],
        interval,
    ))

    # OI abs + pct (Registry column is option_oi, not oi)
    stages.append(_diff_stage(
        ["option_oi"],
        [
            _hz(300.0, column="oi_change_5m"),
            _hz(900.0, column="oi_change_15m"),
            _hz(60.0, column="oi_velocity_1m"),
        ],
        interval,
    ))
    stages.append(_return_stage(
        ["option_oi"],
        [
            _hz(60.0, column="oi_change_1m"),
            _hz(60.0, column="oi_change_pct_1m"),
            _hz(300.0, column="oi_change_pct_5m"),
            _hz(900.0, column="oi_change_pct_15m"),
        ],
        interval,
        scale=100.0,
    ))

    # Volume abs + pct (Registry column is option_day_volume, not volume)
    stages.append(_diff_stage(
        ["option_day_volume"],
        [
            _hz(300.0, column="volume_change_5m"),
            _hz(900.0, column="volume_change_15m"),
        ],
        interval,
    ))
    stages.append(_return_stage(
        ["option_day_volume"],
        [
            _hz(15.0, column="volume_change_15s"),
            _hz(30.0, column="volume_change_30s"),
            _hz(60.0, column="volume_change_1m"),
        ],
        interval,
        scale=100.0,
    ))

    # ATM straddle
    stages.append(_diff_stage(
        ["atm_straddle"],
        [
            _hz(60.0, column="atm_straddle_change_1m"),
            _hz(300.0, column="atm_straddle_change_5m"),
        ],
        interval,
    ))
    stages.append(_return_stage(
        ["atm_straddle"],
        [
            _hz(60.0, column="atm_straddle_change_pct_1m"),
            _hz(300.0, column="atm_straddle_change_pct_5m"),
        ],
        interval,
        scale=100.0,
    ))

    # IV
    stages.append(_diff_stage(
        ["current_iv"],
        [
            _hz(60.0, column="iv_change_1m"),
            _hz(300.0, column="iv_change_5m"),
            _hz(900.0, column="iv_change_15m"),
        ],
        interval,
    ))
    stages.append(_return_stage(
        ["current_iv"],
        [_hz(60.0, column="iv_pct_change_1m")],
        interval,
        scale=100.0,
    ))

    # LTP / spot returns
    stages.append(_return_stage(
        ["ltp"],
        [
            _hz(15.0, column="ltp_return_15s"),
            _hz(30.0, column="ltp_return_30s"),
            _hz(60.0, column="ltp_return_1m"),
        ],
        interval,
        scale=100.0,
    ))
    stages.append(_return_stage(
        ["spot"],
        [
            _hz(15.0, column="spot_change_15s"),
            _hz(30.0, column="spot_change_30s"),
            _hz(60.0, column="spot_change_1m"),
        ],
        interval,
        scale=100.0,
    ))

    # Volume return (frac) for ltp × volume interaction
    stages.append(_return_stage(
        ["option_day_volume"],
        [
            _hz(30.0, column="volume_return_frac_30s"),
            _hz(60.0, column="volume_return_frac_1m"),
            _hz(180.0, column="volume_return_frac_3m"),
            _hz(300.0, column="volume_return_frac_5m"),
            _hz(900.0, column="volume_return_frac_15m"),
        ],
        interval,
        scale=1.0,
        denom_eps=1e-9,
    ))

    # LTP step for spread-normalized packaging
    stages.append(_diff_stage(
        ["ltp"],
        [_hz(float(interval), column="ltp_step")],
        interval,
    ))

    return stages


def _phase2_stages(interval: float) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = [
        {
            "id": "rolling_statistics",
            "enabled": True,
            "order": 35,
            "params": {
                "features": ["current_iv"],
                "windows": [
                    _hz(60.0, column="iv_zscore_1m"),
                    _hz(300.0, column="iv_zscore_5m"),
                    _hz(900.0, column="iv_zscore_15m"),
                    _hz(1800.0, column="iv_zscore_30m"),
                ],
                "stat": "zscore",
                "ddof": 0,
                "partition_by": list(_PARTS),
                "sample_interval_sec": float(interval),
            },
        },
        {
            "id": "rolling_statistics",
            "enabled": True,
            "order": 35,
            "params": {
                "features": ["atm_straddle"],
                "windows": [_hz(1800.0, column="atm_straddle_zscore_30m")],
                "stat": "zscore",
                "ddof": 0,
                "partition_by": list(_PARTS),
                "sample_interval_sec": float(interval),
            },
        },
        {
            "id": "rolling_ohlc",
            "enabled": True,
            "order": 36,
            "params": {
                "features": ["spot"],
                "windows": [_hz(300.0, suffix="5m")],
                "outputs": ["dist_high_pct", "dist_low_pct", "range_pos"],
                "range_eps": 1e-9,
                "column_map": {
                    "dist_high_pct": "spot_dist_high_5m_pct",
                    "dist_low_pct": "spot_dist_low_5m_pct",
                    "range_pos": "spot_range_pos_5m",
                },
                "partition_by": list(_PARTS),
                "sample_interval_sec": float(interval),
            },
        },
        {
            "id": "difference_clip",
            "enabled": True,
            "order": 37,
            "params": {
                "features": ["option_day_volume"],
                "horizons": [
                    _hz(15.0, column="opt_volume_flow_15s"),
                    _hz(30.0, column="opt_volume_flow_30s"),
                    _hz(60.0, column="opt_volume_flow_1m"),
                ],
                "clip_min": 0.0,
                "partition_by": list(_PARTS),
                "sample_interval_sec": float(interval),
            },
        },
        {
            "id": "derived",
            "enabled": True,
            "order": 38,
            "params": {
                "outputs": [
                    {
                        "feature": "atm_straddle",
                        "column": "atm_straddle_slope_5m",
                        "terms": [
                            {"seconds": 0, "coeff": 1.0 / 5.0},
                            {"seconds": 300, "coeff": -1.0 / 5.0},
                        ],
                    },
                    {
                        "feature": "atm_straddle",
                        "column": "atm_straddle_slope_15m",
                        "terms": [
                            {"seconds": 0, "coeff": 1.0 / 15.0},
                            {"seconds": 900, "coeff": -1.0 / 15.0},
                        ],
                    },
                    {
                        "feature": "atm_straddle",
                        "column": "atm_straddle_change_accel",
                        "terms": [
                            {"seconds": 0, "coeff": 0.8},
                            {"seconds": 60, "coeff": -1.0},
                            {"seconds": 300, "coeff": 0.2},
                        ],
                    },
                ],
                "partition_by": list(_PARTS),
                "sample_interval_sec": float(interval),
            },
        },
        {
            "id": "anchor_return",
            "enabled": True,
            "order": 39,
            "params": {
                "outputs": [{
                    "feature": "atm_straddle",
                    "column": "atm_straddle_pct_change_from_open",
                }],
                "scale": 100.0,
                "partition_by": ["trading_day", "token"],
                "sample_interval_sec": float(interval),
            },
        },
        # Depends on atm_straddle_zscore_30m from rolling_statistics above.
        _diff_stage(
            ["atm_straddle_zscore_30m"],
            [_hz(300.0, column="atm_straddle_zscore_change_5m")],
            interval,
        ),
    ]
    stages[-1]["order"] = 40
    return stages


def _ratio(num: str, den: str, output: str, *, eps: float | None = None) -> dict[str, Any]:
    return _pair(num, den, "divide", output, eps=eps)


def _mul(a: str, b: str, output: str) -> dict[str, Any]:
    return _pair(a, b, "multiply", output)


def _sub(a: str, b: str, output: str) -> dict[str, Any]:
    return _pair(a, b, "subtract", output)


def _add(a: str, b: str, output: str) -> dict[str, Any]:
    return _pair(a, b, "add", output)


def _packaging_pairs() -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []

    # Wave 2 — level / ltp|spot
    for n in (9, 20, 50, 100, 200, 300):
        pairs.append(_ratio(f"ltp_ema{n}", "ltp", f"ltp_ema{n}_to_ltp_ratio"))
        pairs.append(_ratio(f"ltp_ema{n}", "spot", f"ltp_ema{n}_to_spot_ratio"))
        pairs.append(_ratio(f"iv_ema{n}", "ltp", f"iv_ema{n}_to_ltp_ratio"))
        pairs.append(_ratio(f"iv_ema{n}", "spot", f"iv_ema{n}_to_spot_ratio"))
        pairs.append(_ratio(f"spot_ema{n}", "ltp", f"spot_ema{n}_to_ltp_ratio"))
    pairs.append(_ratio("ltp_std20", "ltp", "ltp_std20_to_ltp_ratio"))
    pairs.append(_ratio("ltp_std20", "spot", "ltp_std20_to_spot_ratio"))
    for n in (20, 50, 100, 200, 300):
        pairs.append(_ratio(f"spot_high_ema{n}", "ltp", f"spot_high_ema{n}_to_ltp_ratio"))
        pairs.append(_ratio(f"spot_low_ema{n}", "ltp", f"spot_low_ema{n}_to_ltp_ratio"))

    # Wave 3
    pairs.append(_ratio("weighted_ltp_ema", "ltp", "weighted_ltp_ema_to_ltp_ratio"))
    pairs.append(_ratio("weighted_spot_ema", "ltp", "weighted_spot_ema_to_ltp_ratio"))
    pairs.append(_ratio("weighted_spot_high_ema", "ltp", "weighted_spot_high_ema_to_ltp_ratio"))
    pairs.append(_ratio("weighted_spot_low_ema", "ltp", "weighted_spot_low_ema_to_ltp_ratio"))
    pairs.append(_ratio(
        "weighted_spot_high_ema", "weighted_spot_low_ema",
        "weighted_spot_high_ema_to_weighted_spot_low_ema",
    ))
    pairs.append(_ratio(
        "weighted_spot_close_ema", "weighted_spot_low_ema",
        "weighted_spot_ema_to_weighted_spot_low_ema",
    ))
    pairs.append(_ratio(
        "weighted_spot_close_ema", "weighted_spot_high_ema",
        "weighted_spot_ema_to_weighted_spot_high_ema",
    ))

    # Wave 4
    for h in ("1m", "3m", "5m", "10m"):
        pairs.append(_ratio(f"spot_up_score_{h}", "ltp", f"spot_up_score_{h}_to_ltp_ratio"))
        pairs.append(_ratio(f"spot_down_score_{h}", "ltp", f"spot_down_score_{h}_to_ltp_ratio"))
        pairs.append(_ratio(
            "ltp", f"spot_up_sample_count_{h}",
            f"ltp_to_{h}_spot_up_sample_count_ratio",
            eps=1e-6,
        ))
        pairs.append(_ratio(
            "ltp", f"spot_down_sample_count_{h}",
            f"ltp_to_{h}_spot_down_sample_count_ratio",
            eps=1e-6,
        ))

    # Wave 5
    for n in (20, 50, 100, 200, 300):
        pairs.append(_ratio(
            "ltp", f"spot_ema{n}_channel_width",
            f"ltp_to_spot_ema{n}_channel_width_ratio",
            eps=1e-6,
        ))

    # Registry normalization ratios (operand names match Feature Registry)
    for left, right, out in (
        ("ltp", "dgt_reiv_pred", "ltp_to_dgt_reiv_ratio"),
        ("ltp", "bs_reiv_pred", "ltp_to_bs_reiv_ratio"),
        ("dgt_reiv_pred", "ltp", "dgt_reiv_to_ltp_ratio"),
        ("bs_reiv_pred", "ltp", "bs_reiv_to_ltp_ratio"),
        ("dgt_reiv_pred", "spot", "dgt_to_spot_ratio"),
        ("bs_reiv_pred", "spot", "bs_to_spot_ratio"),
        ("spot_rv_5m", "spot_rv_10m", "spot_rv_ratio"),
        ("opt_rv_5m", "opt_rv_10m", "opt_rv_ratio"),
        ("ce_atm6_ltp_sum", "spot", "ce_atm6_ltp_to_spot_ratio"),
        ("pe_atm6_ltp_sum", "spot", "pe_atm6_ltp_to_spot_ratio"),
        ("ce_atm6_ltp_sum", "pe_atm6_ltp_sum", "ce_pe_atm6_ltp_ratio"),
    ):
        pairs.append(_ratio(left, right, out, eps=1e-9 if "atm6" in out else None))
    pairs.append(_add("ce_atm6_ltp_sum", "pe_atm6_ltp_sum", "atm6_total_ltp_sum"))
    pairs.append(_ratio("atm6_total_ltp_sum", "ltp", "atm6_total_to_ltp_ratio"))
    pairs.append(_ratio("atm6_total_ltp_sum", "spot", "atm6_total_to_spot_ratio"))

    # Intermediate used by registry interaction products (not itself pipeline-owned)
    pairs.append(_ratio("spot", "ltp", "spot_to_ltp_ratio"))

    # Wave 6 % packaging
    pairs.append(_sub("spot", "spot_ema20", "spot_minus_spot_ema20"))
    pairs.append(_pair("spot_minus_spot_ema20", "spot_ema20", "divide", "spot_vs_ema20_pct", scale=100.0))
    pairs.append(_sub("spot_ema9", "spot_ema20", "ema9_minus_ema20"))
    pairs.append(_pair("ema9_minus_ema20", "spot_ema20", "divide", "ema_spread_pct", scale=100.0))
    pairs.append(_pair("ema9_minus_ema20", "spot", "divide", "ema_spread_vs_spot_pct", scale=100.0))
    pairs.append(_sub("ce_atm6_ltp_sum", "pe_atm6_ltp_sum", "ce_minus_pe_atm6_for_pct"))
    pairs.append(_add("ce_atm6_ltp_sum", "pe_atm6_ltp_sum", "ce_plus_pe_atm6_for_pct"))
    pairs.append(_ratio(
        "ce_minus_pe_atm6_for_pct", "ce_plus_pe_atm6_for_pct",
        "ce_pe_atm6_ltp_diff_pct",
        eps=1e-9,
    ))

    # Option VWAP / futures packaging
    pairs.append(_sub("ltp", "option_vwap", "ltp_minus_option_vwap"))
    pairs.append(_ratio("ltp_minus_option_vwap", "option_vwap", "ltp_minus_option_vwap_div_option_vwap"))
    pairs.append(_sub("futures_ltp", "futures_vwap", "futures_ltp_minus_futures_vwap"))
    pairs.append(_ratio(
        "futures_ltp_minus_futures_vwap", "futures_vwap",
        "futures_ltp_minus_futures_vwap_div_futures_vwap",
    ))
    pairs.append(_sub("spot", "futures_ltp", "spot_minus_futures_ltp"))
    pairs.append(_sub("spot", "futures_vwap", "spot_minus_futures_vwap"))
    pairs.append(_ratio("spot", "futures_ltp", "spot_div_futures_ltp"))
    pairs.append(_ratio("futures_ltp", "spot", "futures_ltp_div_spot"))

    # Wave A
    pairs.append(_ratio("ltp_step", "bid_ask_spread", "ltp_step_div_bid_ask_spread", eps=1e-9))

    # Greek × ltp_to_spot_ratio products
    pairs.append(_mul("delta", "ltp_to_spot_ratio", "delta_ltp_to_spot_ratio"))
    pairs.append(_mul("gamma", "ltp_to_spot_ratio", "gamma_ltp_to_spot_ratio"))
    pairs.append(_mul("abs_delta", "ltp_to_spot_ratio", "moneyness_delta_ltp_to_spot_ratio"))

    # Registry interaction products (depend on packaging above + zscores)
    pairs.append(_mul("delta", "spot", "delta_x_spot"))
    pairs.append(_mul("gamma", "spot", "gamma_x_spot"))

    # weighted_iv_zscore is not a Registry column; skip that historical product for now.
    pairs.append(_mul("weighted_spot_ema_to_ltp_ratio", "delta", "weighted_spot_ema_to_ltp_ratio_x_delta"))
    for h in ("1m", "5m", "15m"):
        pairs.append(_mul(
            "weighted_spot_ema_to_ltp_ratio", f"iv_zscore_{h}",
            f"weighted_spot_ema_to_ltp_ratio_x_iv_zscore_{h}",
        ))
        pairs.append(_mul(
            f"weighted_spot_ema_to_ltp_ratio_x_iv_zscore_{h}", "delta",
            f"weighted_spot_ema_to_ltp_ratio_x_iv_zscore_{h}_x_delta",
        ))

    for n in (9, 20, 50, 100, 200, 300):
        pairs.append(_mul(f"ltp_ema{n}_to_spot_ratio", f"iv_ema{n}", f"ltp_ema{n}_to_spot_ratio_x_iv_ema{n}"))
        pairs.append(_mul("spot_to_ltp_ratio", f"iv_ema{n}", f"spot_to_ltp_ratio_x_iv_ema{n}"))
        pairs.append(_mul(
            f"spot_to_ltp_ratio_x_iv_ema{n}", "moneyness",
            f"spot_to_ltp_ratio_x_iv_ema{n}_x_moneyness",
        ))
        pairs.append(_mul(f"spot_ema{n}_to_ltp_ratio", "moneyness", f"spot_ema{n}_to_ltp_ratio_x_moneyness"))
        pairs.append(_mul(f"ltp_ema{n}_to_spot_ratio", "moneyness", f"ltp_ema{n}_to_spot_ratio_x_moneyness"))

    pairs.append(_mul("weighted_spot_ema_to_ltp_ratio", "moneyness", "weighted_spot_ema_to_ltp_ratio_x_moneyness"))
    pairs.append(_mul(
        "weighted_spot_ema_to_ltp_ratio_x_moneyness", "delta",
        "weighted_spot_ema_to_ltp_ratio_x_moneyness_x_delta",
    ))

    # ltp × volume return
    for suf, col in (
        ("30s", "volume_return_frac_30s"),
        ("1m", "volume_return_frac_1m"),
        ("3m", "volume_return_frac_3m"),
        ("5m", "volume_return_frac_5m"),
        ("15m", "volume_return_frac_15m"),
    ):
        pairs.append(_mul("ltp", col, f"ltp_x_volume_change_pct_{suf}"))

    return pairs


def build_pipeline_features_transformation_config(
    *,
    sample_interval_sec: float = 3.0,
    exclude_features: frozenset[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Return a transformation_config that recreates Pipeline Features."""
    interval = float(sample_interval_sec)
    stages = _time_shift_stages(interval) + _phase2_stages(interval)
    base = normalize_transformation_config({
        "transformation_pipeline_version": 1,
        "transformations": stages,
    })
    # spot_to_ltp_ratio is Registry Base — used by many products.
    pairs = _packaging_pairs()
    cfg = merge_interaction_into_config(
        base,
        enabled=True,
        pairs=pairs,
        div_zero="null",
        eps=1e-12,
    )
    # Soften duplicate / missing for analysis builds (overwrite Master leftovers).
    for t in cfg.get("transformations") or []:
        if str((t or {}).get("id") or "") == "interaction":
            params = dict(t.get("params") or {})
            params["fail_on_duplicate_output"] = False
            params["overwrite"] = True
            t["params"] = params
    if exclude_features:
        cfg = prune_pipeline_transformation_config(cfg, exclude_features)
    return cfg


def prune_pipeline_transformation_config(
    config: dict[str, Any],
    exclude_features: frozenset[str] | set[str] | Sequence[str],
) -> dict[str, Any]:
    """Drop stages / windows / pairs that emit permanently deleted features.

    Handles ``horizons``, ``windows`` (rolling_statistics / rolling_ohlc),
    interaction outputs, and derived/anchor ``column`` outputs. Stages whose
    only remaining work is retired are removed entirely.
    """
    skip = {str(n).strip() for n in exclude_features if str(n).strip()}
    if not skip:
        return config
    out = dict(config)
    kept_stages: list[dict[str, Any]] = []

    def _filter_timed(items: list[Any]) -> list[Any]:
        kept: list[Any] = []
        for item in items:
            if (
                isinstance(item, dict)
                and str(item.get("column") or "").strip() in skip
            ):
                continue
            kept.append(item)
        return kept

    for raw in list(out.get("transformations") or []):
        stage = dict(raw or {})
        sid = str(stage.get("id") or "")
        params = dict(stage.get("params") or {})
        if sid == "interaction":
            pairs = []
            for p in list(params.get("pairs") or []):
                if not isinstance(p, dict):
                    continue
                pair_out = str(p.get("output") or "").strip()
                if pair_out in skip:
                    continue
                left = str(p.get("left") or "").strip()
                right = str(p.get("right") or "").strip()
                if left in skip or right in skip:
                    continue
                pairs.append(p)
            if not pairs:
                continue
            params["pairs"] = pairs
            stage["params"] = params
            kept_stages.append(stage)
            continue

        # rolling_statistics / rolling_ohlc / exponential use ``windows``.
        windows = list(params.get("windows") or [])
        if windows and any(isinstance(w, dict) and w.get("column") for w in windows):
            kept_w = _filter_timed(windows)
            if not kept_w:
                continue
            params["windows"] = kept_w
            # Drop retired names from column_map values when present.
            cmap = params.get("column_map")
            if isinstance(cmap, dict) and cmap:
                params["column_map"] = {
                    k: v
                    for k, v in cmap.items()
                    if str(v or "").strip() not in skip
                }
            stage["params"] = params
            kept_stages.append(stage)
            continue

        horizons = list(params.get("horizons") or [])
        if horizons and any(isinstance(h, dict) and h.get("column") for h in horizons):
            kept_h = _filter_timed(horizons)
            if not kept_h:
                continue
            params["horizons"] = kept_h
            # Difference of a retired pipeline base (e.g. zscore → change).
            features = [
                str(f).strip()
                for f in (params.get("features") or [])
                if str(f).strip()
            ]
            if features:
                kept_feats = [f for f in features if f not in skip]
                if not kept_feats:
                    continue
                params["features"] = kept_feats
            stage["params"] = params
            kept_stages.append(stage)
            continue

        outputs = params.get("outputs")
        if isinstance(outputs, list) and outputs:
            kept_out: list[Any] = []
            for item in outputs:
                if isinstance(item, dict):
                    name = str(
                        item.get("name")
                        or item.get("output")
                        or item.get("column")
                        or ""
                    ).strip()
                    base = str(item.get("feature") or "").strip()
                    if name and name in skip:
                        continue
                    if base and base in skip:
                        continue
                    kept_out.append(item)
                else:
                    name = str(item or "").strip()
                    if name and name in skip:
                        continue
                    kept_out.append(item)
            if not kept_out:
                continue
            params["outputs"] = kept_out
            stage["params"] = params
            kept_stages.append(stage)
            continue

        # Feature lists on lag/diff/return: drop retired sources; remove stage if empty.
        features = [str(f).strip() for f in (params.get("features") or []) if str(f).strip()]
        if features:
            kept_feats = [f for f in features if f not in skip]
            if not kept_feats:
                continue
            if kept_feats != features:
                params["features"] = kept_feats
                stage["params"] = params
        kept_stages.append(stage)

    out["transformations"] = kept_stages
    return out


def expected_pipeline_outputs_from_config(config: dict[str, Any] | None) -> list[str]:
    """Best-effort list of output column names the config intends to create."""
    from .feature_migration import PIPELINE_OWNED_FEATURES
    from .transformations import describe_pipeline

    if not config:
        return sorted(PIPELINE_OWNED_FEATURES)
    try:
        plan = describe_pipeline(config)
        names: list[str] = []
        for stage in getattr(plan, "stages", []) or []:
            for out in getattr(stage, "outputs", []) or []:
                name = getattr(out, "name", None) or (out.get("name") if isinstance(out, dict) else None)
                if name:
                    names.append(str(name))
        # Prefer intersection with catalogue when available.
        owned = set(PIPELINE_OWNED_FEATURES)
        owned_hits = [n for n in names if n in owned]
        return owned_hits or names
    except Exception:
        return sorted(PIPELINE_OWNED_FEATURES)


__all__ = [
    "build_pipeline_features_transformation_config",
    "expected_pipeline_outputs_from_config",
    "prune_pipeline_transformation_config",
]
