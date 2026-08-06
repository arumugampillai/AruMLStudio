"""Layer 2 — live feature engine publishing immutable FeatureSnapshot."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Mapping

from .progress import InferenceProgressReporter

import pandas as pd

from chain_replay_ml.replay_feature_scoring import build_replay_day_frame
from chain_replay_ml.replay_scoring_cache import (
    get_cached_shared_features,
    set_cached_shared_features,
    shared_inference_cache_key,
)

from .market_state import LiveMarketState
from .snapshot import FeatureSnapshot
from .versions import feature_version, market_state_version
from .warmup import READINESS_PROBE_FEATURES, assess_feature_readiness

_INTERNAL_ROW_KEYS = frozenset({
    "_opt_tl", "_spot", "_atm", "_feature_raw",
})

# Row identity columns — not model features; excluded from FeatureSnapshot counts.
_FEATURE_ROW_METADATA = frozenset({
    "trading_day", "market", "expiry", "timestamp", "token", "symbol", "option_type",
})

# Warmup: longest grid-bar stack is EMA200 on feature_grid_step_sec (see feature_grid_policy).
_WARMUP_LOOKBACK_SEC = 900.0
_WARMUP_EMA_BARS = 200


def count_union_features_built(features: Mapping[str, Any], union_features: list[str]) -> int:
    """Count model-registry features present in a built feature dict."""
    return sum(1 for name in union_features if name in features)


def feature_signature(feature_names: list[str]) -> str:
    joined = "|".join(sorted(set(str(f) for f in feature_names if f)))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def warmup_required_sec(replay_config: dict[str, Any]) -> float:
    """Legacy helper — session warmup uses WARMUP_MINUTES from constants."""
    del replay_config
    from chain_replay_ml.constants import WARMUP_MINUTES
    return float(WARMUP_MINUTES * 60)


def _series_to_feature_dict(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in row.items():
        key_s = str(key)
        if key in _INTERNAL_ROW_KEYS or key_s.startswith("_") or key_s in _FEATURE_ROW_METADATA:
            continue
        if isinstance(val, (int, float, str, bool)) or val is None:
            out[str(key)] = val
        elif pd.notna(val):
            try:
                out[str(key)] = float(val)
            except (TypeError, ValueError):
                out[str(key)] = val
    return out


def _feature_row_for_token(
    df: pd.DataFrame,
    token: str,
    grid_ts: float,
    *,
    eps: float = 0.05,
) -> pd.Series | None:
    if df.empty or not token:
        return None
    sub = df[df["token"].astype(str) == str(token)]
    if sub.empty:
        return None
    ts_col = sub["timestamp"].astype(float)
    exact = sub[ts_col.sub(grid_ts).abs() <= eps]
    if not exact.empty:
        return exact.iloc[0]
    prior = sub[ts_col <= grid_ts + eps]
    if not prior.empty:
        return prior.sort_values("timestamp").iloc[-1]
    nearest_idx = (ts_col - grid_ts).abs().idxmin()
    return sub.loc[nearest_idx]


class LiveFeatureEngine:
    """Build one shared feature dictionary per grid timestamp."""

    READINESS_PROBES = READINESS_PROBE_FEATURES

    @staticmethod
    def is_ready(features: Mapping[str, Any]) -> bool:
        return bool(assess_feature_readiness(features)["warmup_complete"])

    @staticmethod
    def readiness_info(features: Mapping[str, Any]) -> dict[str, object]:
        return assess_feature_readiness(features)

    def build_snapshot(
        self,
        *,
        data_dir: str,
        state: LiveMarketState,
        token: str,
        grid_ts: float,
        union_features: list[str],
        replay_config: dict[str, Any],
        expiry_hint: str | None = None,
        progress: InferenceProgressReporter | None = None,
    ) -> tuple[FeatureSnapshot | None, str | None]:
        sig = feature_signature(union_features)
        date_str = state.trading_day
        cache_key = shared_inference_cache_key(data_dir, date_str, expiry_hint, grid_ts, token, sig)
        cached = get_cached_shared_features(cache_key)

        t0 = time.perf_counter()
        if cached is not None:
            features = cached
            build_sec = 0.0
            if progress:
                progress.feature_build_begin()
                progress.mark_all_groups_done()
        else:
            if progress:
                progress.feature_build_begin()

            market = state.market
            include = frozenset({str(token)}) if token else None

            from chain_replay_ml.dataset_builder.schema_registry import load_feature_registry
            from chain_replay_ml.replay_feature_scoring import merge_replay_feature_build_plan

            registry = load_feature_registry()
            enabled = list(
                replay_config.get("feature_groups_implemented") or replay_config.get("feature_groups") or []
            )
            enabled, _impl, _pending, per_group = merge_replay_feature_build_plan(
                enabled, registry, union_features,
            )
            group_labels = {
                gid: str((registry.get("groups") or {}).get(gid, {}).get("label") or gid)
                for gid in enabled
            }
            label_to_gid = {group_labels[gid]: gid for gid in enabled if gid in per_group}

            def on_step_progress(step_name: str, detail: str) -> None:
                if not progress:
                    return
                if step_name == "load_day_context":
                    progress.set_detail(detail or "Loading option chain from tick DB…")
                elif step_name in ("build_rows", "chain_maps"):
                    progress.set_detail(detail or "Preparing feature inputs…")
                elif step_name == "build_day_rows":
                    progress.set_detail(detail or "Building feature groups…")
                elif step_name == "to_dataframe":
                    progress.set_detail(detail or "Finalizing feature row…")

            def on_group_start(gid: str, label: str) -> None:
                if progress:
                    progress.set_detail(f"Feature group: {label}")
                    progress.group_start(gid, label)

            def on_group_progress(label: str, current: int, total: int) -> None:
                if not progress:
                    return
                gid = label_to_gid.get(label)
                if not gid:
                    return
                feat_total = len(per_group.get(gid) or [])
                if feat_total > 0 and total > 0:
                    feat_done = min(feat_total, max(0, int(round(current * feat_total / total))))
                    if current > 0 and feat_done == 0:
                        feat_done = 1
                else:
                    feat_done = current
                    feat_total = total
                progress.group_progress(gid, feat_done, feat_total)
                progress.set_detail(f"{label} · {feat_done}/{feat_total} features")

            def on_group_done(gid: str) -> None:
                if progress:
                    progress.group_done(gid)

            df, err, _expiry_resolution, _stats = build_replay_day_frame(
                data_dir,
                replay_config,
                date_str,
                expiry_hint=expiry_hint or state.expiry,
                target="",
                underlying=market,
                required_features=union_features,
                inference_only=True,
                only_timestamp=grid_ts,
                include_tokens=include,
                token_only=bool(include),
                enrich_tokens_only=include,
                day_context=state.ctx,
                on_step_progress=on_step_progress if progress else None,
                on_feature_group_start=on_group_start if progress else None,
                on_feature_group_progress=on_group_progress if progress else None,
                on_feature_group_done=on_group_done if progress else None,
            )
            if err:
                return None, err

            row = _feature_row_for_token(df, token, grid_ts)
            if row is None:
                return None, "no_feature_row"

            features = _series_to_feature_dict(row)
            if not features:
                return None, "empty_feature_dict"

            set_cached_shared_features(cache_key, features)
            build_sec = round(time.perf_counter() - t0, 3)

        warmup_done = self.is_ready(features)

        snapshot = FeatureSnapshot.create(
            timestamp=float(grid_ts),
            token=str(token),
            features=features,
            feature_version=feature_version(),
            market_state_version=market_state_version(),
            trading_day=date_str,
            expiry=str(expiry_hint or state.expiry),
            source=state.source_kind,
            warmup_complete=warmup_done,
            build_sec=build_sec,
            feature_sig=sig,
        )
        return snapshot, None
