"""IV- and options-aware market regime classification."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .regime_detection import _num, analyze_regimes

REGIME_FEATURE_COLUMNS = (
    "current_iv",
    "chain_pcr",
    "atm_pcr",
    "delta",
    "theta",
    "spot_ema20_to_ltp_ratio",
    "ltp",
)


def load_validation_feature_map(
    data_dir: str,
    *,
    run: dict[str, Any],
    fold: dict[str, Any],
) -> dict[int, dict[str, Any]] | None:
    """Map prediction row_index → feature values for validation fold."""
    from .feature_rehydration import slice_feature_matrix

    cols = list(REGIME_FEATURE_COLUMNS)
    sliced = slice_feature_matrix(data_dir, run=run, fold=fold)
    if not sliced.get("ok"):
        return None
    val: pd.DataFrame = sliced["validation"]
    available = [c for c in cols if c in val.columns]
    if not available:
        return None
    out: dict[int, dict[str, Any]] = {}
    for i in range(len(val)):
        row = val.iloc[i]
        out[i] = {c: (round(float(row[c]), 6) if pd.notna(row[c]) else None) for c in available}
    return out


def _classify_iv_row(
    row: dict[str, Any],
    feats: dict[str, Any] | None,
    *,
    prev_spot: float | None,
    prev_iv: float | None,
    prev_pcr: float | None,
) -> list[str]:
    tags: list[str] = []
    tags.extend(_classify_spot_row(row, prev_spot))

    if not feats:
        return tags

    iv = _num(feats.get("current_iv"))
    if iv is not None and prev_iv is not None and prev_iv > 0:
        chg = (iv - prev_iv) / prev_iv * 100.0
        if chg > 2.0:
            tags.append("IV Expansion")
        elif chg < -2.0:
            tags.append("IV Compression")

    pcr = _num(feats.get("chain_pcr")) or _num(feats.get("atm_pcr"))
    if pcr is not None and prev_pcr is not None:
        if pcr > prev_pcr * 1.01:
            tags.append("PCR Rising")
        elif pcr < prev_pcr * 0.99:
            tags.append("PCR Falling")

    ltp = _num(row.get("ltp")) or _num(feats.get("ltp"))
    if ltp is not None:
        if ltp >= 35:
            tags.append("ATM Premium High")
        elif ltp < 18:
            tags.append("ATM Premium Low")

    delta = _num(feats.get("delta"))
    if delta is not None:
        if abs(delta) > 0.45:
            tags.append("Gamma Zone")
        if abs(delta) < 0.15:
            tags.append("Theta Zone")

    theta = _num(feats.get("theta"))
    if theta is not None and abs(theta) > 0.5:
        tags.append("High Theta")

    ts = _num(row.get("timestamp"))
    if ts is not None:
        sec = int(ts) % 86400
        if sec < 3600 * 10 + 900:
            tags.append("Expiry Morning")
        elif sec < 3600 * 14:
            tags.append("Expiry Afternoon")
        elif sec >= 3600 * 15:
            tags.append("Expiry Closing Hour")

    return list(dict.fromkeys(tags))


def _classify_spot_row(row: dict[str, Any], prev_spot: float | None) -> list[str]:
    tags: list[str] = []
    spot = _num(row.get("spot"))
    if spot is not None and prev_spot is not None and prev_spot > 0:
        ret = (spot - prev_spot) / prev_spot * 100.0
        if ret > 0.15:
            tags.append("Momentum")
        elif ret < -0.15:
            tags.append("Reversal")
        if abs(ret) > 0.35:
            tags.append("Trending")
        elif abs(ret) < 0.03:
            tags.append("Range")
    ltp = _num(row.get("ltp"))
    if ltp is not None and ltp > 40:
        tags.append("High Premium")
    return tags


def analyze_iv_regimes(
    rows: list[dict[str, Any]],
    feature_map: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify rows with IV/PCR/premium regimes and compute MAE per regime."""
    if not rows:
        return {"available": False, "regimes": [], "note": "No rows."}

    basic = analyze_regimes(rows)
    if not feature_map:
        basic["iv_enriched"] = False
        basic["note"] = (basic.get("note") or "") + " IV/PCR regimes need master dataset features."
        return basic

    sorted_rows = sorted(rows, key=lambda r: (_num(r.get("timestamp")) or 0.0))
    by_regime: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    prev_spot: float | None = None
    prev_iv: float | None = None
    prev_pcr: float | None = None

    for row in sorted_rows:
        ri = row.get("row_index")
        feats = feature_map.get(int(ri)) if ri is not None else None
        tags = _classify_iv_row(row, feats, prev_spot=prev_spot, prev_iv=prev_iv, prev_pcr=prev_pcr)
        if not tags:
            tags = ["Neutral"]
        primary = tags[0]
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1
        err = _num(row.get("prediction_error"))
        if err is not None:
            for tag in tags:
                by_regime.setdefault(tag, []).append(abs(err))

        spot = _num(row.get("spot"))
        if spot is not None:
            prev_spot = spot
        if feats:
            iv = _num(feats.get("current_iv"))
            pcr = _num(feats.get("chain_pcr")) or _num(feats.get("atm_pcr"))
            if iv is not None:
                prev_iv = iv
            if pcr is not None:
                prev_pcr = pcr

    regime_rows: list[dict[str, Any]] = []
    for regime, errors in sorted(by_regime.items(), key=lambda kv: -len(kv[1])):
        regime_rows.append({
            "regime": regime,
            "row_count": counts.get(regime, 0),
            "mae": round(sum(errors) / len(errors), 4) if errors else None,
        })

    return {
        "available": True,
        "iv_enriched": True,
        "volatility_regime": basic.get("volatility_regime"),
        "volatility_proxy_pct": basic.get("volatility_proxy_pct"),
        "regimes": regime_rows,
        "note": "IV-aware regimes from master dataset features (current_iv, chain_pcr, delta, theta, EMA ratio).",
    }
