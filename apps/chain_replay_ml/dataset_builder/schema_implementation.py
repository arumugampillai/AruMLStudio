"""Map schema formula_ref values to source module and function."""

from __future__ import annotations

from typing import Any

# formula_ref or feature name → implementation location
_OVERRIDES: dict[str, dict[str, str]] = {
    "bs_delta": {"module": "chain_replay_ml/bs.py", "function": "bs_delta()"},
    "bs_gamma": {"module": "chain_replay_ml/bs.py", "function": "bs_gamma()"},
    "bs_theta": {"module": "chain_replay_ml/bs.py", "function": "bs_theta()"},
    "bs_vega": {"module": "chain_replay_ml/bs.py", "function": "bs_vega()"},
    "implied_vol": {"module": "chain_replay_ml/bs.py", "function": "implied_volatility()"},
    "future_ltp_at_horizon": {
        "module": "chain_replay_ml/dataset_builder/feature_plugins.py",
        "function": "horizon_column_name()",
    },
}

_GROUP_DEFAULTS: dict[str, dict[str, str]] = {
    "price": {
        "module": "chain_replay_ml/features_atm_band.py",
        "function": "extract_timeline_features()",
    },
    "market_microstructure": {
        "module": "chain_replay_ml/dataset_builder/market_microstructure.py",
        "function": "enrich_market_microstructure_features()",
    },
    "momentum": {
        "module": "chain_replay_ml/features_atm_band.py",
        "function": "extract_timeline_features()",
    },
    "sharp_momentum": {
        "module": "chain_replay_ml/dataset_builder/sharp_momentum.py",
        "function": "enrich_sharp_momentum_features()",
    },
    "iv_zscore": {
        "module": "chain_replay_ml/dataset_builder/iv_zscore_features.py",
        "function": "enrich_iv_zscore_features()",
    },
    "iv_ema_ratio": {
        "module": "chain_replay_ml/dataset_builder/iv_ema_ratio_features.py",
        "function": "enrich_iv_ema_ratio_features()",
    },
    "historic_spot_ema": {
        "module": "chain_replay_ml/dataset_builder/historic_spot_ema_context.py",
        "function": "enrich_historic_spot_ema_features()",
    },
    "spot_hl": {
        "module": "chain_replay_ml/dataset_builder/spot_hl_registry.py",
        "function": "enrich_spot_hl_ratio_registry_features() / enrich_spot_hl_composite_registry_features()",
    },
    "chain_flow": {
        "module": "chain_replay_ml/dataset_builder/current_to_atm6_flow.py",
        "function": "enrich_current_to_atm6_flow_features()",
    },
    "advanced": {
        "module": "chain_replay_ml/dataset_builder/extended_features.py",
        "function": "enrich_dataset_features()",
    },
    "moneyness": {
        "module": "chain_replay_ml/features_atm_band.py",
        "function": "extract_timeline_features()",
    },
    "ltp_to_spot": {
        "module": "chain_replay_ml/features_atm_band.py",
        "function": "ltp_to_spot_ratio_lag_change_features() — change_* only; lag_* is Pipeline Owned",
    },
    "ltp_to_others": {
        "module": "chain_replay_ml/dataset_builder/extended_features.py",
        "function": "enrich_with_chain_maps()",
    },
    "spot_and_other_ratio": {
        "module": "chain_replay_ml/dataset_builder/extended_features.py",
        "function": "enrich_with_chain_maps()",
    },
    "time": {
        "module": "chain_replay_ml/features_atm_band.py",
        "function": "extract_timeline_features()",
    },
    "volume": {
        "module": "chain_replay_ml/features_atm_band.py",
        "function": "extract_timeline_features()",
    },
    "historical": {
        "module": "chain_replay_ml/features_atm_band.py",
        "function": "extract_ohlc_features_for_timeline()",
    },
    "greeks": {
        "module": "chain_replay_ml/bs.py",
        "function": "greeks()",
    },
    "iv": {
        "module": "chain_replay_ml/dataset_builder/extended_features.py",
        "function": "enrich_dataset_features()",
    },
    "dgt_reiv": {
        "module": "chain_replay_ml/dataset_builder/extended_features.py",
        "function": "enrich_dataset_features()",
    },
    "ratio": {
        "module": "chain_replay_ml/dataset_builder/extended_features.py",
        "function": "enrich_dataset_features()",
    },
    "oi": {
        "module": "chain_replay_ml/dataset_builder/extended_features.py",
        "function": "enrich_dataset_features()",
    },
    "advanced": {
        "module": "chain_replay_ml/dataset_builder/extended_features.py",
        "function": "enrich_dataset_features()",
    },
    "chain": {
        "module": "chain_replay_ml/dataset_builder/chain_maps.py / chain_iv_skew.py",
        "function": "precompute_chain_maps() / chain_features_at()",
    },
    "atm_straddle": {
        "module": "chain_replay_ml/dataset_builder/chain_maps.py",
        "function": "chain_features_at()",
    },
    "atm6_ltp": {
        "module": "chain_replay_ml/dataset_builder/chain_maps.py",
        "function": "chain_features_at()",
    },
}


def resolve_implementation(
    name: str,
    formula_ref: str | None = None,
    group_id: str = "",
) -> dict[str, str]:
    ref = str(formula_ref or name or "").strip()
    if ref in _OVERRIDES:
        return dict(_OVERRIDES[ref])
    if ref.startswith("dgt_") or ref in (
        "ltp_to_dgt_reiv_ratio", "ltp_to_bs_reiv_ratio",
        "dgt_reiv_to_ltp_ratio", "bs_reiv_to_ltp_ratio",
        "dgt_to_spot_ratio", "bs_to_spot_ratio",
    ):
        return {
            "module": "chain_replay_ml/dataset_builder/extended_features.py",
            "function": "enrich_dataset_features()",
        }
    if ref in (
        "side_to_ltp_ratio", "atm6_total_to_ltp_ratio",
        "ltp_ema9_to_ltp_ratio", "ltp_ema20_to_ltp_ratio",
        "ltp_ema50_to_ltp_ratio", "ltp_ema100_to_ltp_ratio", "ltp_ema200_to_ltp_ratio",
        "ltp_std20_to_ltp_ratio",
        "spot_ema9_to_ltp_ratio", "spot_ema20_to_ltp_ratio",
        "spot_ema50_to_ltp_ratio", "spot_ema100_to_ltp_ratio", "spot_ema200_to_ltp_ratio",
        "atm6_total_to_spot_ratio",
        "ltp_ema9_to_spot_ratio", "ltp_ema20_to_spot_ratio",
        "ltp_ema50_to_spot_ratio", "ltp_ema100_to_spot_ratio", "ltp_ema200_to_spot_ratio",
        "ltp_std20_to_spot_ratio",
    ):
        return {
            "module": "chain_replay_ml/dataset_builder/extended_features.py",
            "function": "enrich_with_chain_maps()",
        }
    if ref.startswith("spot_ema") and ref.endswith("_to_ltp_ratio_x_moneyness"):
        return {
            "module": "chain_replay_ml/dataset_builder/spot_ratio_moneyness_features.py",
            "function": "enrich_spot_ratio_moneyness_features()",
        }
    if ref.startswith("ltp_ema") and ref.endswith("_to_spot_ratio_x_moneyness"):
        return {
            "module": "chain_replay_ml/dataset_builder/spot_ratio_moneyness_features.py",
            "function": "enrich_spot_ratio_moneyness_features()",
        }
    if ref == "spot_ema300_to_ltp_ratio":
        return {
            "module": "chain_replay_ml/dataset_builder/spot_ratio_moneyness_features.py",
            "function": "enrich_spot_ratio_moneyness_features()",
        }
    if ref.startswith("bs_"):
        return {"module": "chain_replay_ml/bs.py", "function": f"{ref}()"}
    if group_id in _GROUP_DEFAULTS:
        return dict(_GROUP_DEFAULTS[group_id])
    fn = ref if ref.endswith("()") else f"{ref}()"
    return {
        "module": "chain_replay_ml/dataset_builder/feature_plugins.py",
        "function": fn,
    }


def implementation_for_column(
    name: str,
    *,
    formula_ref: str | None = None,
    group_id: str = "",
    doc: dict[str, Any] | None = None,
) -> dict[str, str]:
    doc = doc or {}
    if doc.get("implementation"):
        return dict(doc["implementation"])
    return resolve_implementation(name, formula_ref, group_id)
