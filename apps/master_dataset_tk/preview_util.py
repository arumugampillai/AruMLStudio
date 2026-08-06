"""Row estimates and master DB status — no HTTP."""

from __future__ import annotations

from typing import Any

from .build_service import chart_data_dir


def estimate_rows_for_sources(
    sources: list[dict[str, Any]],
    *,
    interval_sec: int,
    horizons_sec: list[int],
    atm_band: int | None = None,
) -> int:
    """Same formula as Create Dataset ``estimatedRowCountForSources``."""
    if not sources:
        return 0
    step_sec = max(int(interval_sec or 10), 1)
    horizons = [int(h) for h in horizons_sec if int(h) > 0] or [300]
    max_horizon = max(horizons)
    usable_sec = 22500 - 60 - max_horizon
    if usable_sec <= 0:
        return 0
    sample_points = usable_sec // step_sec + 1
    band = max(0, int(atm_band if atm_band is not None else 10))
    strikes_per_ts = (2 * band + 1) * 2
    return sample_points * strikes_per_ts * len(sources)


def read_master_status(
    chart_dir: str,
    *,
    market: str,
    interval_sec: int,
) -> dict[str, Any] | None:
    try:
        from chain_replay_ml.dataset_builder.master_status import read_master_dataset_status

        return read_master_dataset_status(
            chart_data_dir(chart_dir),
            market=str(market).upper(),
            interval_sec=int(interval_sec),
        )
    except Exception:
        return None


def day_in_master(status: dict[str, Any] | None, trading_day: str) -> bool:
    if not status:
        return False
    days = status.get("days_in_master") or []
    return str(trading_day) in days


def day_row_count(status: dict[str, Any] | None, trading_day: str) -> int | None:
    if not status:
        return None
    counts = status.get("row_counts_by_day") or {}
    val = counts.get(trading_day)
    return int(val) if val is not None else None
