"""Frame backend — Arrow → Polars standard path + Pandas adapters (Phase P1).

See docs/architecture/POLARS_MIGRATION.md
"""

from __future__ import annotations

from .convert import (
    BRIDGE_ARROW_PANDAS,
    BRIDGE_ARROW_POLARS,
    BRIDGE_ARROW_POLARS_PANDAS,
    arrow_table_to_pandas,
    arrow_table_to_polars,
    polars_to_pandas,
    require_polars,
)
from .write import (
    BRIDGE_WRITE_ARROW_PANDAS,
    BRIDGE_WRITE_POLARS,
    META_TEXT_COLS,
    coerce_frame_for_parquet,
    frame_to_arrow_table_via_polars,
    write_parquet_via_polars,
)
from .studio_stats import (
    distribution_summary_via_polars,
    feature_distribution_rows_via_polars,
)

__all__ = [
    "BRIDGE_ARROW_PANDAS",
    "BRIDGE_ARROW_POLARS",
    "BRIDGE_ARROW_POLARS_PANDAS",
    "BRIDGE_WRITE_ARROW_PANDAS",
    "BRIDGE_WRITE_POLARS",
    "META_TEXT_COLS",
    "arrow_table_to_pandas",
    "arrow_table_to_polars",
    "coerce_frame_for_parquet",
    "distribution_summary_via_polars",
    "feature_distribution_rows_via_polars",
    "frame_to_arrow_table_via_polars",
    "polars_to_pandas",
    "require_polars",
    "write_parquet_via_polars",
]
