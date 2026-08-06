"""
Bridge Kotak ``TickRingStore`` / ``TickSample`` history to replay ``TickTimeline``.

Backtest ML uses ``chain_replay_ml.ticks.TickTimeline`` for point-in-time LTP/OI
lookup (``ltp_rupees_at``, ``volume_at``, ``oi_at``). Live NeoApp stores ticks in
fixed-size rings. This module builds replay-compatible timelines without touching
the main GUI loop or signal engine.

Volume: option cumulative volume comes from the chart server tick log
(``day_volume`` via ``/replay/{token}/ticks``) when available; rings supply
live LTP/OI fallback and index timelines.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from api.tick_ring import TickRingStore, TickSample

if TYPE_CHECKING:
    from chain_replay_ml.ticks import TickTimeline

_CHART_DIR = Path(__file__).resolve().parents[2] / "angelone" / "chart"
_replay_tick_timeline_cls: type | None = None


def replay_tick_timeline_class() -> type:
    """Lazy import of ``chain_replay_ml.ticks.TickTimeline`` (chart tree on sys.path)."""
    global _replay_tick_timeline_cls
    if _replay_tick_timeline_cls is not None:
        return _replay_tick_timeline_cls
    chart_dir = str(_CHART_DIR)
    if chart_dir not in sys.path:
        sys.path.insert(0, chart_dir)
    from chain_replay_ml.ticks import TickTimeline as _TickTimeline

    _replay_tick_timeline_cls = _TickTimeline
    return _TickTimeline


def _ltp_paise_from_sample(sample: TickSample) -> int | None:
    try:
        paise = int(round(float(sample.ltp) * 100.0))
    except (TypeError, ValueError):
        return None
    return paise if paise > 0 else None


def samples_to_tick_timeline(
    samples: Iterable[TickSample],
    *,
    include_volume: bool = False,
) -> TickTimeline:
    """
    Convert ring samples to a sorted replay ``TickTimeline``.

    Duplicate timestamps keep the last sample at that ts. Out-of-order samples are
    sorted before append.
    """
    TickTimeline = replay_tick_timeline_class()
    timeline = TickTimeline()
    merged: dict[float, tuple[int, int, int]] = {}

    for sample in samples:
        paise = _ltp_paise_from_sample(sample)
        if paise is None:
            continue
        ts = float(sample.ts)
        vol = 0
        if include_volume and sample.ltq is not None:
            try:
                vol = max(0, int(sample.ltq))
            except (TypeError, ValueError):
                vol = 0
        oi = 0
        if sample.oi is not None:
            try:
                oi = max(0, int(sample.oi))
            except (TypeError, ValueError):
                oi = 0
        merged[ts] = (paise, vol, oi)

    for ts in sorted(merged):
        paise, vol, oi = merged[ts]
        timeline.append(ts, paise, volume=vol, oi=oi, spread_paise=0)
    return timeline


def ring_to_tick_timeline(
    store: TickRingStore,
    key: str,
    *,
    last_n: int | None = None,
    include_volume: bool = False,
) -> TickTimeline:
    """Build a ``TickTimeline`` from one token/index key in the tick ring store."""
    samples = store.snapshot(_ring_store_lookup_key(key), last_n=last_n)
    return samples_to_tick_timeline(samples, include_volume=include_volume)


def _ring_store_lookup_key(key: str) -> str:
    k = str(key or "").strip()
    if k.isdigit():
        return k
    try:
        from research.atm_band_ml.band_evaluator import index_ring_store_key

        return index_ring_store_key(k)
    except ImportError:
        return k


def chart_api_ticks_to_timeline(
    payload: Mapping[str, Any],
    *,
    open_ts: float | None = None,
    close_ts: float | None = None,
) -> TickTimeline:
    """
    Convert chart server ``/replay/{token}/ticks`` JSON to a replay ``TickTimeline``.

    Row shape: ``[ts_seconds, ltp_paise, vol_today, ltq, seq]``.
    ``vol_today`` is cumulative session volume (same as replay ``day_volume``).
    """
    TickTimeline = replay_tick_timeline_class()
    timeline = TickTimeline()
    rows = payload.get("ticks") or []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        try:
            ts = float(row[0])
            ltp_paise = int(row[1])
        except (TypeError, ValueError):
            continue
        if ltp_paise <= 0:
            continue
        if open_ts is not None and ts < float(open_ts) - 1.0:
            continue
        if close_ts is not None and ts > float(close_ts) + 1.0:
            continue
        vol = 0
        if len(row) > 2 and row[2] is not None:
            try:
                vol = max(0, int(row[2]))
            except (TypeError, ValueError):
                vol = 0
        timeline.append(ts, ltp_paise, volume=vol, oi=0, spread_paise=0)
    return timeline


def merge_chart_volume_timeline(
    ring_tl: TickTimeline,
    chart_tl: TickTimeline,
) -> TickTimeline:
    """
    Prefer chart session ticks (LTP + cumulative volume); overlay OI from ring.

    Falls back to ring-only when chart has no rows.
    """
    if chart_tl is None or not chart_tl.timestamps:
        return ring_tl
    TickTimeline = replay_tick_timeline_class()
    merged = TickTimeline()
    for i, ts in enumerate(chart_tl.timestamps):
        paise = chart_tl.ltps_paise[i]
        vol = chart_tl.volumes[i] if i < len(chart_tl.volumes) else 0
        oi = ring_tl.oi_at(ts)
        if oi is None:
            oi = 0
        merged.append(ts, paise, volume=int(vol or 0), oi=int(oi), spread_paise=0)
    return merged


def index_timeline_with_chart(
    store: TickRingStore,
    index_key: str,
    chart_payload: Mapping[str, Any] | None,
    *,
    open_ts: float | None = None,
    close_ts: float | None = None,
) -> TickTimeline:
    """Session index timeline: chart-server ticks + recent ring overlay."""
    ring_tl = ring_to_tick_timeline(store, str(index_key).strip())
    if not chart_payload:
        return ring_tl
    chart_tl = chart_api_ticks_to_timeline(
        chart_payload,
        open_ts=open_ts,
        close_ts=close_ts,
    )
    if not chart_tl.timestamps:
        return ring_tl
    TickTimeline = replay_tick_timeline_class()
    merged = TickTimeline()
    for i, ts in enumerate(chart_tl.timestamps):
        vol = chart_tl.volumes[i] if i < len(chart_tl.volumes) else 0
        merged.append(ts, chart_tl.ltps_paise[i], volume=int(vol or 0), oi=0, spread_paise=0)
    if ring_tl.timestamps:
        last_chart_ts = float(chart_tl.timestamps[-1])
        for i, ts in enumerate(ring_tl.timestamps):
            if float(ts) + 0.25 < last_chart_ts:
                continue
            vol = ring_tl.volumes[i] if i < len(ring_tl.volumes) else 0
            oi = ring_tl.oi_list[i] if i < len(ring_tl.oi_list) else 0
            merged.append(
                float(ts),
                ring_tl.ltps_paise[i],
                volume=int(vol or 0),
                oi=int(oi or 0),
                spread_paise=0,
            )
    return merged


def option_timeline_with_chart_volume(
    store: TickRingStore,
    token: str,
    chart_payload: Mapping[str, Any] | None,
    *,
    open_ts: float | None = None,
    close_ts: float | None = None,
) -> TickTimeline:
    """Ring timeline enriched with chart-server cumulative volume when available."""
    ring_tl = ring_to_tick_timeline(store, str(token).strip())
    if not chart_payload:
        return ring_tl
    chart_tl = chart_api_ticks_to_timeline(
        chart_payload,
        open_ts=open_ts,
        close_ts=close_ts,
    )
    return merge_chart_volume_timeline(ring_tl, chart_tl)


def ring_ltp_rupees_at(samples: Sequence[TickSample], target_ts: float) -> float | None:
    """Point-in-time LTP: newest sample with ``ts <= target_ts`` (works on unsorted rings)."""
    if not samples:
        return None
    target = float(target_ts)
    best: TickSample | None = None
    for sample in samples:
        if float(sample.ts) > target:
            continue
        try:
            px = float(sample.ltp)
        except (TypeError, ValueError):
            continue
        if px <= 0:
            continue
        if best is None or float(sample.ts) > float(best.ts):
            best = sample
    return float(best.ltp) if best is not None else None


def timeline_ltp_rupees_at(timeline: TickTimeline, target_ts: float) -> float | None:
    return timeline.ltp_rupees_at(float(target_ts))


def compare_ring_vs_timeline(
    samples: Sequence[TickSample],
    probe_times: Sequence[float],
    *,
    include_volume: bool = False,
) -> list[dict[str, object]]:
    """
    Parity check: ring bisect lookup vs ``TickTimeline.ltp_rupees_at`` per probe ts.

    Returns rows with ``match`` True when both are None or within 0.005 rupees.
    """
    timeline = samples_to_tick_timeline(samples, include_volume=include_volume)
    rows: list[dict[str, object]] = []
    for target_ts in probe_times:
        ring_px = ring_ltp_rupees_at(samples, target_ts)
        tl_px = timeline_ltp_rupees_at(timeline, target_ts)
        if ring_px is None and tl_px is None:
            match = True
        elif ring_px is None or tl_px is None:
            match = False
        else:
            match = abs(float(ring_px) - float(tl_px)) < 0.005
        rows.append(
            {
                "target_ts": float(target_ts),
                "ring_ltp": ring_px,
                "timeline_ltp": tl_px,
                "match": match,
            }
        )
    return rows


def parity_all_match(rows: Sequence[dict[str, object]]) -> bool:
    return bool(rows) and all(bool(r.get("match")) for r in rows)
