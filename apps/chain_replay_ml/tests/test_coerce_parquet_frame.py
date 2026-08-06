"""Parquet coercion must stay consistent across chunked export batches."""

from __future__ import annotations

import unittest

import pandas as pd
import pyarrow as pa

from chain_replay_ml.dataset_builder.writer import _coerce_parquet_frame


class TestCoerceParquetFrame(unittest.TestCase):
    def test_all_null_numeric_column_matches_later_chunk(self) -> None:
        cols = ["trading_day", "token", "future_ltp_5s", "ltp"]
        chunk_a = pd.DataFrame(
            [
                {"trading_day": "2026-05-27", "token": "1", "future_ltp_5s": None, "ltp": 20.0},
                {"trading_day": "2026-05-27", "token": "2", "future_ltp_5s": None, "ltp": 21.0},
            ],
            columns=cols,
        )
        chunk_b = pd.DataFrame(
            [
                {"trading_day": "2026-05-27", "token": "3", "future_ltp_5s": 22.5, "ltp": 23.0},
            ],
            columns=cols,
        )
        sch_a = pa.Table.from_pandas(_coerce_parquet_frame(chunk_a), preserve_index=False).schema
        sch_b = pa.Table.from_pandas(_coerce_parquet_frame(chunk_b), preserve_index=False).schema
        for name in sch_a.names:
            self.assertEqual(
                sch_a.field(name).type,
                sch_b.field(name).type,
                msg=f"dtype mismatch for {name}",
            )


if __name__ == "__main__":
    unittest.main()
