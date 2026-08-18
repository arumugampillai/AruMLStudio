"""Canonical Dataset Context derivation and resolution for Feature Recommendation Evidence."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

LEGACY_UNKNOWN_CONTEXT_ID = "legacy_unknown"


@dataclass(frozen=True)
class DatasetContext:
    context_id: str
    market: str
    sampling_interval_sec: int
    sampling_label: str
    sliding_window: str
    feature_project_id: str
    context_key: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_context_key(
    *,
    market: str,
    sampling_interval_sec: int,
    sliding_window: str = "standard",
    feature_project_id: str = "all",
) -> str:
    m = str(market or "NIFTY").strip().upper()
    sec = max(1, int(sampling_interval_sec or 3))
    win = str(sliding_window or "standard").strip().lower()
    fpid = str(feature_project_id or "all").strip().lower()
    return f"{m}:{sec}:{win}:{fpid}"


def generate_context_id(context_key: str) -> str:
    if context_key == LEGACY_UNKNOWN_CONTEXT_ID:
        return LEGACY_UNKNOWN_CONTEXT_ID
    digest = hashlib.sha256(context_key.encode("utf-8")).hexdigest()[:12]
    return f"ctx_{digest}"


def build_dataset_context(
    *,
    market: str,
    sampling_interval_sec: int,
    sliding_window: str = "standard",
    feature_project_id: str = "all",
    sampling_label: str | None = None,
    created_at: str | None = None,
) -> DatasetContext:
    m = str(market or "NIFTY").strip().upper()
    sec = max(1, int(sampling_interval_sec or 3))
    s_label = str(sampling_label or f"{sec}s").strip().lower()
    win = str(sliding_window or "standard").strip().lower()
    fpid = str(feature_project_id or "all").strip().lower()
    key = build_context_key(
        market=m,
        sampling_interval_sec=sec,
        sliding_window=win,
        feature_project_id=fpid,
    )
    cid = generate_context_id(key)
    return DatasetContext(
        context_id=cid,
        market=m,
        sampling_interval_sec=sec,
        sampling_label=s_label,
        sliding_window=win,
        feature_project_id=fpid,
        context_key=key,
        created_at=created_at or _utc_now(),
    )


def resolve_context_from_model_package(
    data_dir: str,
    model_name: str,
) -> DatasetContext | None:
    """Read model package config / dataset_metadata to recover canonical Dataset Context."""
    from chain_replay_ml.training.paths import model_package_dir, safe_model_name

    pkg = model_package_dir(data_dir, safe_model_name(model_name))
    cfg_path = os.path.join(pkg, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception:
        return None

    if not isinstance(cfg, dict):
        return None

    dataset_meta = cfg.get("dataset_metadata") or {}
    market = dataset_meta.get("market") or cfg.get("market") or "NIFTY"
    interval = (
        dataset_meta.get("sampling_interval_sec")
        or cfg.get("sampling_interval_sec")
        or 3
    )
    s_label = dataset_meta.get("sampling_label") or f"{interval}s"
    
    # Extract sliding window / ATM band
    strike_sel = dataset_meta.get("strike_selection") or {}
    if isinstance(strike_sel, dict) and strike_sel.get("atm_band"):
        sliding_window = f"atm_{strike_sel.get('atm_band')}"
    else:
        sliding_window = "standard"

    fpid = cfg.get("feature_project_id") or dataset_meta.get("feature_project_id") or "all"

    return build_dataset_context(
        market=str(market),
        sampling_interval_sec=int(interval),
        sampling_label=str(s_label),
        sliding_window=str(sliding_window),
        feature_project_id=str(fpid),
    )


def resolve_context_or_legacy(
    data_dir: str,
    model_name: str,
) -> DatasetContext:
    """Resolve real context if model exists, otherwise return legacy_unknown context."""
    ctx = resolve_context_from_model_package(data_dir, model_name)
    if ctx is not None:
        return ctx
    return DatasetContext(
        context_id=LEGACY_UNKNOWN_CONTEXT_ID,
        market="UNKNOWN",
        sampling_interval_sec=0,
        sampling_label="unknown",
        sliding_window="unknown",
        feature_project_id="unknown",
        context_key=LEGACY_UNKNOWN_CONTEXT_ID,
        created_at=_utc_now(),
    )
