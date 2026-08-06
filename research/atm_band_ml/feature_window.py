"""Diagnostics for index feature-window completeness (live debug)."""
from __future__ import annotations

import math
from typing import Any

from research.atm_band_ml.feature_builder import (
    FEATURE_BAR_SEC,
    FEATURE_LOOKBACKS_SEC,
    _bucket_has_tick,
    index_ltp_rupees_at,
    missing_5s_buckets,
)


def feature_window_diagnostics(
    ts: float,
    index_timeline,
    *,
    open_ts: float,
    close_ts: float,
    missing_buckets: set[float] | None = None,
    live: bool = False,
) -> dict[str, Any]:
    """Explain which lookback bucket blocks ``is_feature_window_complete``."""
    target = float(ts)
    if index_timeline is None or not index_timeline.timestamps:
        return {
            "complete": False,
            "reason": "no_index_timeline",
            "lookbacks": [],
            "missing_bucket_count": None,
            "index_tick_count": 0,
            "first_tick_ts": None,
            "last_tick_ts": None,
        }
    if missing_buckets is None:
        missing_buckets = missing_5s_buckets(
            open_ts,
            close_ts,
            index_timeline,
            as_of_ts=target,
        )
    lookbacks: list[dict[str, Any]] = []
    complete = True
    reason = ""
    for lb in FEATURE_LOOKBACKS_SEC:
        bucket = math.floor((target - lb) / FEATURE_BAR_SEC) * FEATURE_BAR_SEC
        ltp = index_ltp_rupees_at(index_timeline, target - lb)
        bucket_has_tick = _bucket_has_tick(index_timeline, bucket)
        bucket_missing = bucket in missing_buckets
        if live:
            ok = ltp is not None
        else:
            ok = (not bucket_missing) and ltp is not None
        if not ok and complete:
            complete = False
            if live:
                reason = f"no_index_ltp@{int(lb)}s"
            elif bucket_missing:
                reason = f"missing_5s_bucket@{int(lb)}s"
            else:
                reason = f"no_index_ltp@{int(lb)}s"
        lookbacks.append(
            {
                "lookback_sec": lb,
                "bucket": bucket,
                "bucket_missing": bucket_missing,
                "bucket_has_tick": bucket_has_tick,
                "ltp": ltp,
                "ok": ok,
            }
        )
    stamps = index_timeline.timestamps
    return {
        "complete": complete,
        "reason": reason or "ok",
        "lookbacks": lookbacks,
        "missing_bucket_count": len(missing_buckets),
        "index_tick_count": len(stamps),
        "first_tick_ts": float(stamps[0]) if stamps else None,
        "last_tick_ts": float(stamps[-1]) if stamps else None,
        "live": live,
    }
