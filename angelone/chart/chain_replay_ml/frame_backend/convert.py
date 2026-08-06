"""Arrow ↔ Polars ↔ Pandas conversion helpers (Phase P1)."""

from __future__ import annotations

from typing import Any

BRIDGE_ARROW_PANDAS = "arrow_pandas"
BRIDGE_ARROW_POLARS = "arrow_polars"
BRIDGE_ARROW_POLARS_PANDAS = "arrow_polars_pandas"


def require_polars() -> Any:
    """Import polars or raise a clear InstallError-style message."""
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "polars is required for the Arrow→Polars frame bridge. "
            "Install with: pip install polars"
        ) from exc
    return pl


def arrow_table_to_polars(table: Any) -> Any:
    """Standard path: Arrow → Polars."""
    pl = require_polars()
    if table is None:
        return pl.DataFrame()
    return pl.from_arrow(table)


def polars_to_pandas(frame: Any) -> Any:
    """Thin adapter: Polars → Pandas (P1/P3 edge)."""
    import pandas as pd

    if frame is None:
        return pd.DataFrame()
    if isinstance(frame, pd.DataFrame):
        return frame
    to_pandas = getattr(frame, "to_pandas", None)
    if callable(to_pandas):
        return to_pandas()
    raise TypeError(f"Cannot convert to pandas: {type(frame)!r}")


def arrow_table_to_pandas(
    table: Any,
    *,
    via_polars: bool = True,
) -> tuple[Any, str]:
    """Materialize an Arrow table to Pandas.

    Default (P1 standard): Arrow → Polars → Pandas.
    Fallback: Arrow → Pandas directly if Polars is unavailable or ``via_polars=False``.

    Returns ``(dataframe, bridge_id)``.
    """
    if table is None:
        import pandas as pd

        return pd.DataFrame(), BRIDGE_ARROW_PANDAS

    if via_polars:
        try:
            pl_df = arrow_table_to_polars(table)
            return polars_to_pandas(pl_df), BRIDGE_ARROW_POLARS_PANDAS
        except ImportError:
            pass

    to_pandas = getattr(table, "to_pandas", None)
    if not callable(to_pandas):
        raise TypeError(f"Arrow table has no to_pandas(): {type(table)!r}")
    return to_pandas(), BRIDGE_ARROW_PANDAS
