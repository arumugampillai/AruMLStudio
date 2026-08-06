"""Parquet write path via Polars (Phase P4)."""

from __future__ import annotations

from typing import Any, Sequence

BRIDGE_WRITE_POLARS = "polars_parquet"
BRIDGE_WRITE_ARROW_PANDAS = "arrow_pandas_parquet"

# Meta columns kept as strings so chunked ParquetWriter schemas stay stable.
META_TEXT_COLS: frozenset[str] = frozenset({
    "trading_day",
    "market",
    "expiry",
    "option_type",
    "symbol",
    "token",
})


def coerce_frame_for_parquet(
    frame: Any,
    *,
    meta_text_cols: Sequence[str] | None = None,
) -> Any:
    """Normalize dtypes for Parquet: meta → Utf8, features → Float64.

    Accepts Pandas or Polars; returns a Polars DataFrame.
    Falls back to a Pandas-shaped Polars frame built via ``pl.from_pandas``.
    """
    from .convert import arrow_table_to_polars, require_polars

    pl = require_polars()
    meta = frozenset(meta_text_cols) if meta_text_cols is not None else META_TEXT_COLS

    if frame is None:
        return pl.DataFrame()

    # Already Polars
    if hasattr(frame, "write_parquet") and hasattr(frame, "with_columns"):
        pl_df = frame
    else:
        try:
            import pyarrow as pa
            import pandas as pd

            if isinstance(frame, pd.DataFrame):
                pl_df = arrow_table_to_polars(
                    pa.Table.from_pandas(frame, preserve_index=False)
                )
            else:
                pl_df = pl.from_pandas(pd.DataFrame(frame))
        except Exception:
            import pandas as pd

            pl_df = pl.from_pandas(
                frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
            )

    if pl_df.width == 0:
        return pl_df

    exprs = []
    for col in pl_df.columns:
        c = pl.col(col)
        if col in meta:
            # Keep nulls as null; stringify everything else (parity with writer._coerce).
            exprs.append(
                pl.when(c.is_null())
                .then(None)
                .otherwise(c.cast(pl.Utf8, strict=False))
                .alias(col)
            )
        else:
            exprs.append(c.cast(pl.Float64, strict=False).alias(col))
    return pl_df.with_columns(exprs)


def write_parquet_via_polars(
    frame: Any,
    parquet_path: str,
    *,
    meta_text_cols: Sequence[str] | None = None,
    compression: str = "zstd",
) -> str:
    """Coerce then write Parquet with Polars. Returns bridge id.

    On Polars failure, falls back to Pandas → Arrow → Parquet.
    """
    try:
        pl_df = coerce_frame_for_parquet(frame, meta_text_cols=meta_text_cols)
        # Native Polars write (Arrow under the hood).
        pl_df.write_parquet(parquet_path, compression=compression)
        return BRIDGE_WRITE_POLARS
    except Exception:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not isinstance(frame, pd.DataFrame):
            from .convert import polars_to_pandas

            try:
                pdf = polars_to_pandas(frame)
            except Exception:
                pdf = pd.DataFrame(frame)
        else:
            pdf = frame
        # Caller may already have coerced; write best-effort.
        table = pa.Table.from_pandas(pdf, preserve_index=False)
        pq.write_table(table, parquet_path, compression=compression)
        return BRIDGE_WRITE_ARROW_PANDAS


def frame_to_arrow_table_via_polars(
    frame: Any,
    *,
    meta_text_cols: Sequence[str] | None = None,
    coerce: bool = True,
) -> Any:
    """Pandas/Polars → (optional coerce) → Arrow Table via Polars."""
    from .convert import require_polars

    pl = require_polars()
    if coerce:
        pl_df = coerce_frame_for_parquet(frame, meta_text_cols=meta_text_cols)
    elif hasattr(frame, "to_arrow") and hasattr(frame, "with_columns"):
        pl_df = frame
    else:
        import pandas as pd
        import pyarrow as pa
        from .convert import arrow_table_to_polars

        if isinstance(frame, pd.DataFrame):
            try:
                pl_df = arrow_table_to_polars(
                    pa.Table.from_pandas(frame, preserve_index=False)
                )
            except Exception:
                pl_df = pl.from_pandas(frame)
        else:
            pl_df = pl.DataFrame(frame)
    to_arrow = getattr(pl_df, "to_arrow", None)
    if callable(to_arrow):
        return to_arrow()
    import pyarrow as pa

    return pa.Table.from_pandas(pl_df.to_pandas(), preserve_index=False)
