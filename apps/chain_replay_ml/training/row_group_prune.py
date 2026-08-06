"""Tail row-group pruning for Create Model parquet loads (Opt 1).

When a dataset exceeds ``MAX_TRAINING_ROWS``, training only needs the chronological
tail. Master exports write row groups in time order, so we can skip early groups
at the Parquet metadata level instead of reading 3M rows and ``df.tail(750k)``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

from .memory_utils import MAX_TRAINING_ROWS


@dataclass(frozen=True)
class TailRowGroupPlan:
    """Which row groups to read for a chronological tail."""

    indices: tuple[int, ...]
    rows_in_plan: int
    total_rows: int
    total_row_groups: int
    max_rows_requested: int
    skip_reason: str | None = None
    chronology_ok: bool = True
    chronology_source: str | None = None

    @property
    def pruned(self) -> bool:
        return self.total_row_groups > 0 and len(self.indices) < self.total_row_groups

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": bool(self.pruned),
            "row_groups_total": self.total_row_groups,
            "row_groups_read": list(self.indices),
            "row_groups_skipped": max(0, self.total_row_groups - len(self.indices)),
            "rows_in_plan": self.rows_in_plan,
            "total_rows": self.total_rows,
            "max_rows_requested": self.max_rows_requested,
            "chronology_ok": self.chronology_ok,
            "chronology_source": self.chronology_source,
            "skip_reason": self.skip_reason,
        }


def row_group_prune_mode() -> str:
    """``auto`` (default) | ``on`` | ``off`` via ``ARUNEO_TRAIN_ROW_GROUP_PRUNE``."""
    raw = str(os.getenv("ARUNEO_TRAIN_ROW_GROUP_PRUNE", "auto") or "auto").strip().lower()
    if raw in ("0", "false", "no", "off", "disable", "disabled"):
        return "off"
    if raw in ("1", "true", "yes", "on", "force"):
        return "on"
    return "auto"


def premium_overread_factor() -> float:
    """When premium filter will drop rows, over-read this many times the cap."""
    raw = str(os.getenv("ARUNEO_TRAIN_ROW_GROUP_OVERREAD", "2") or "2").strip()
    try:
        val = float(raw)
    except ValueError:
        return 2.0
    return max(1.0, min(val, 20.0))


def target_rows_for_prune(*, premium_filter: bool, max_rows: int = MAX_TRAINING_ROWS) -> int:
    """Rows to cover via row groups before premium filter + final cap."""
    base = max(1, int(max_rows))
    if premium_filter:
        return int(base * premium_overread_factor())
    return base


def _metadata_claims_chronological(metadata: dict[str, Any] | None) -> bool:
    meta = metadata if isinstance(metadata, dict) else {}
    if meta.get("is_sorted") is True:
        return True
    row_order = meta.get("row_order")
    if isinstance(row_order, (list, tuple)):
        keys = [str(x) for x in row_order]
        if keys[:1] == ["trading_day"] or keys[:2] == ["trading_day", "timestamp"]:
            return True
    return False


def _stat_min_max(rg_meta: Any, column: str) -> tuple[Any, Any] | None:
    try:
        for i in range(rg_meta.num_columns):
            col = rg_meta.column(i)
            path = str(getattr(col, "path_in_schema", "") or "")
            if path != column:
                continue
            stats = col.statistics
            if stats is None:
                return None
            has_mm = getattr(stats, "has_min_max", None)
            if callable(has_mm):
                if not has_mm():
                    return None
            elif has_mm is False:
                return None
            return stats.min, stats.max
    except Exception:
        return None
    return None


def _row_groups_chronological(parquet_path: str) -> tuple[bool, str | None]:
    """True when row-group min(timestamp|trading_day) is non-decreasing."""
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(parquet_path)
        meta = pf.metadata
        if meta is None or meta.num_row_groups <= 1:
            return True, "single_or_empty"
        schema_names = set(pf.schema_arrow.names) if pf.schema_arrow is not None else set()
        prefer = [c for c in ("timestamp", "trading_day") if c in schema_names]
        if not prefer:
            return False, None
        for col in prefer:
            mins: list[Any] = []
            ok = True
            for i in range(meta.num_row_groups):
                mm = _stat_min_max(meta.row_group(i), col)
                if mm is None:
                    ok = False
                    break
                mins.append(mm[0])
            if not ok or len(mins) < 2:
                continue
            non_decreasing = all(mins[i] <= mins[i + 1] for i in range(len(mins) - 1))
            if non_decreasing:
                return True, f"stats:{col}"
            return False, f"stats:{col}:not_monotonic"
        return False, None
    except Exception:
        return False, None


def plan_tail_row_groups(
    parquet_path: str,
    *,
    max_rows: int = MAX_TRAINING_ROWS,
    metadata: dict[str, Any] | None = None,
    mode: str | None = None,
) -> TailRowGroupPlan | None:
    """Select trailing row groups that cover at least ``max_rows`` newest rows.

    Returns ``None`` when pruning is disabled or unsafe / unnecessary.
    """
    resolved_mode = (mode or row_group_prune_mode()).strip().lower()
    if resolved_mode == "off":
        return None

    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(parquet_path)
        meta = pf.metadata
    except Exception:
        return None
    if meta is None:
        return None

    total_rows = int(meta.num_rows)
    n_rg = int(meta.num_row_groups)
    if n_rg <= 0 or total_rows <= 0:
        return None
    if total_rows <= int(max_rows):
        return TailRowGroupPlan(
            indices=tuple(range(n_rg)),
            rows_in_plan=total_rows,
            total_rows=total_rows,
            total_row_groups=n_rg,
            max_rows_requested=int(max_rows),
            skip_reason="below_cap",
            chronology_ok=True,
            chronology_source="n/a",
        )

    meta_ok = _metadata_claims_chronological(metadata)
    chrono_ok, chrono_src = _row_groups_chronological(parquet_path)
    if resolved_mode == "auto":
        if not (meta_ok or chrono_ok):
            return TailRowGroupPlan(
                indices=tuple(range(n_rg)),
                rows_in_plan=total_rows,
                total_rows=total_rows,
                total_row_groups=n_rg,
                max_rows_requested=int(max_rows),
                skip_reason="chronology_unproven",
                chronology_ok=False,
                chronology_source=chrono_src,
            )
    elif resolved_mode == "on" and not (meta_ok or chrono_ok):
        return TailRowGroupPlan(
            indices=tuple(range(n_rg)),
            rows_in_plan=total_rows,
            total_rows=total_rows,
            total_row_groups=n_rg,
            max_rows_requested=int(max_rows),
            skip_reason="chronology_unproven",
            chronology_ok=False,
            chronology_source=chrono_src,
        )

    selected: list[int] = []
    accumulated = 0
    for rg_idx in reversed(range(n_rg)):
        rg_rows = int(meta.row_group(rg_idx).num_rows)
        selected.append(rg_idx)
        accumulated += rg_rows
        if accumulated >= int(max_rows):
            break
    selected_sorted = tuple(sorted(selected))
    return TailRowGroupPlan(
        indices=selected_sorted,
        rows_in_plan=accumulated,
        total_rows=total_rows,
        total_row_groups=n_rg,
        max_rows_requested=int(max_rows),
        skip_reason=None if len(selected_sorted) < n_rg else "all_groups_needed",
        chronology_ok=True,
        chronology_source=("metadata" if meta_ok else chrono_src),
    )


def read_parquet_row_groups(
    parquet_path: str,
    row_groups: Sequence[int],
    *,
    columns: Sequence[str] | None = None,
):
    """Read selected row groups (column-pruned) as a PyArrow table."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    cols = list(columns) if columns else None
    return pf.read_row_groups(list(row_groups), columns=cols)
