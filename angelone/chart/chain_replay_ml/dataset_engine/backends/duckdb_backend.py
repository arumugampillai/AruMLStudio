"""DuckDB backend — first Dataset Engine adapter (ADR)."""

from __future__ import annotations

import time
from typing import Any

from ..planner import QueryPlan
from ..types import ExecutionStats, QueryResult


class DuckDbBackend:
    """Execute plans via DuckDB ``read_parquet`` (Phase 1)."""

    name = "duckdb"

    def __init__(self, *, parquet_path: str | None = None) -> None:
        self._parquet_path = parquet_path

    def execute(self, plan: QueryPlan) -> QueryResult:
        t0 = time.perf_counter()
        path = self._parquet_path or _resolve_parquet_path(plan.dataset_id)
        if not path:
            raise FileNotFoundError(
                f"DuckDbBackend: cannot resolve parquet for dataset_id={plan.dataset_id!r}"
            )
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DuckDbBackend requires pyarrow for schema validation."
            ) from exc

        schema = pq.read_schema(path)
        schema_names = set(schema.names)
        if plan.columns:
            missing = [c for c in plan.columns if c not in schema_names]
            if missing:
                raise KeyError(
                    "Dataset Engine schema validation failed: "
                    f"missing columns {missing} (dataset={plan.dataset_id!r})"
                )
        _validate_filter_columns(plan.filters, schema_names)

        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DuckDbBackend requires duckdb. "
                "Install it before using the Dataset Engine query path."
            ) from exc

        cols_sql = "*"
        if plan.columns:
            cols_sql = ", ".join(f'"{c}"' for c in plan.columns)

        where_sql, params = _filters_to_sql(plan.filters)
        bind: list[Any] = []
        row_groups_read: list[int] | None = None
        source_sql = "read_parquet(?)"
        arrow_source = None

        # Create Model Opt-1: materialize only trailing Parquet row groups, then
        # apply filters / ORDER BY in DuckDB over the in-memory Arrow table.
        if plan.row_groups is not None:
            rg_list = [int(i) for i in plan.row_groups]
            if not rg_list:
                raise ValueError("QueryPlan.row_groups is empty")
            n_rg = int(pq.ParquetFile(path).metadata.num_row_groups)
            if any(i < 0 or i >= n_rg for i in rg_list):
                raise ValueError(
                    f"QueryPlan.row_groups out of range for {n_rg} groups: {rg_list}"
                )
            read_cols = list(plan.columns) if plan.columns else None
            arrow_source = pq.ParquetFile(path).read_row_groups(rg_list, columns=read_cols)
            source_sql = "_rg_prune"
            row_groups_read = rg_list
        else:
            bind.append(path)

        sql = f"SELECT {cols_sql} FROM {source_sql} "
        if where_sql:
            sql += f"WHERE {where_sql} "
            bind.extend(params)

        if plan.order_by:
            # Keep only columns present in the parquet / SELECT list.
            available = set(schema.names)
            if plan.columns:
                available &= set(plan.columns)
            order_cols = [c for c in plan.order_by if c in available]
            if order_cols:
                sql += "ORDER BY " + ", ".join(f'"{c}"' for c in order_cols) + " "

        # Phase-1 sampling stand-in: LIMIT only (deterministic cap). Replace with
        # reservoir / TABLESAMPLE when Analysis parity requires random draws.
        sample_mode = "none"
        if plan.sample and plan.sample.max_rows is not None:
            sql += "LIMIT ? "
            bind.append(int(plan.sample.max_rows))
            sample_mode = "limit"
        sql = sql.rstrip() + ";"

        con = duckdb.connect(database=":memory:")
        try:
            if arrow_source is not None:
                con.register("_rg_prune", arrow_source)
            relation = con.execute(sql, bind) if bind else con.execute(sql)
            table = relation.to_arrow_table()
        finally:
            con.close()

        elapsed = time.perf_counter() - t0
        stats = ExecutionStats(
            rows_returned=table.num_rows,
            rows_scanned=None,
            partitions_scanned=1 if plan.partitions is None else len(plan.partitions),
            partitions_pruned=plan.partitions_pruned,
            columns_read=tuple(plan.columns) if plan.columns else tuple(table.column_names),
            execution_time_sec=round(elapsed, 6),
            bytes_read=None,
            backend=self.name,
            extra={
                "dataset_id": plan.dataset_id,
                "sample_seed": None if plan.sample is None else plan.sample.seed,
                "sample_mode": sample_mode,
                "order_by": list(plan.order_by or ()),
                "row_groups": row_groups_read,
                "plan_notes": list(plan.notes),
            },
        )
        return QueryResult(table=table, stats=stats)


def _validate_filter_columns(filters: dict[str, Any], schema_names: set[str]) -> None:
    """Fail early if a filter implies a column absent from the parquet schema."""
    needed: list[str] = []
    if filters.get("trading_days") is not None or filters.get("trading_day") is not None:
        needed.append("trading_day")
    if filters.get("start_day") is not None or filters.get("end_day") is not None:
        needed.append("trading_day")
    if any(k in filters for k in ("ltp_min", "ltp_max", "premium_min", "premium_max")):
        needed.append("ltp")
    if any(k in filters for k in ("dte_max", "days_to_expiry_max")):
        needed.append("days_to_expiry")
    if filters.get("atm_distance_max") is not None:
        needed.append("atm_distance")
    missing = [c for c in needed if c not in schema_names]
    if missing:
        raise KeyError(
            "Dataset Engine schema validation failed: "
            f"filters require missing columns {missing}"
        )


def _resolve_parquet_path(dataset_id: str) -> str | None:
    """Resolve dataset_id to a parquet file path.

    Phase-1: if ``dataset_id`` is already a filesystem path ending in .parquet,
    use it. Registry lookup hooks land with Model Builder wiring.
    """
    import os

    raw = str(dataset_id or "").strip()
    if raw.lower().endswith(".parquet") and os.path.isfile(raw):
        return os.path.abspath(raw)
    return None


def _filters_to_sql(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    """Compile a small set of structured filters to SQL.

    Supported keys (Phase 1):
      - trading_day / trading_days
      - ltp_min / ltp_max  (or premium_min / premium_max → column ``ltp``)
      - dte_max / days_to_expiry_max
      - atm_distance_max
    Unknown keys are ignored with no error (planner may warn later).
    """
    clauses: list[str] = []
    params: list[Any] = []

    if "trading_days" in filters and filters["trading_days"] is not None:
        days = list(filters["trading_days"])
        if days:
            placeholders = ", ".join("?" for _ in days)
            clauses.append(f'"trading_day" IN ({placeholders})')
            params.extend(str(d) for d in days)
    elif filters.get("trading_day") is not None:
        clauses.append('"trading_day" = ?')
        params.append(str(filters["trading_day"]))

    start_day = filters.get("start_day")
    end_day = filters.get("end_day")
    if start_day is not None:
        clauses.append('"trading_day" >= ?')
        params.append(str(start_day))
    if end_day is not None:
        clauses.append('"trading_day" <= ?')
        params.append(str(end_day))

    lo = filters.get("ltp_min", filters.get("premium_min"))
    hi = filters.get("ltp_max", filters.get("premium_max"))
    if lo is not None:
        clauses.append('"ltp" >= ?')
        params.append(float(lo))
    if hi is not None:
        clauses.append('"ltp" <= ?')
        params.append(float(hi))

    dte_max = filters.get("dte_max", filters.get("days_to_expiry_max"))
    if dte_max is not None:
        # Column name may be days_to_expiry or dte — prefer days_to_expiry.
        clauses.append('"days_to_expiry" <= ?')
        params.append(float(dte_max))

    atm_max = filters.get("atm_distance_max")
    if atm_max is not None:
        clauses.append('"atm_distance" <= ?')
        params.append(float(atm_max))

    return " AND ".join(clauses), params
