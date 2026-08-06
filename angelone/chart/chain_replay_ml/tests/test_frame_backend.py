"""Unit tests for Arrow → Polars → Pandas frame bridge (Phase P1)."""

from __future__ import annotations

import unittest


class FrameBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import polars  # noqa: F401
            import pyarrow as pa  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_arrow_to_polars_to_pandas_roundtrip(self) -> None:
        import pandas as pd
        import pyarrow as pa

        from chain_replay_ml.frame_backend import (
            BRIDGE_ARROW_POLARS_PANDAS,
            arrow_table_to_pandas,
            arrow_table_to_polars,
        )

        table = pa.table({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
        pl_df = arrow_table_to_polars(table)
        self.assertEqual(pl_df.shape, (3, 2))

        pdf, bridge = arrow_table_to_pandas(table, via_polars=True)
        self.assertEqual(bridge, BRIDGE_ARROW_POLARS_PANDAS)
        self.assertIsInstance(pdf, pd.DataFrame)
        self.assertEqual(list(pdf.columns), ["a", "b"])
        self.assertEqual(len(pdf), 3)
        self.assertEqual(int(pdf["a"].sum()), 6)

    def test_legacy_arrow_pandas_fallback(self) -> None:
        import pandas as pd
        import pyarrow as pa

        from chain_replay_ml.frame_backend import (
            BRIDGE_ARROW_PANDAS,
            arrow_table_to_pandas,
        )

        table = pa.table({"x": [1, 2]})
        pdf, bridge = arrow_table_to_pandas(table, via_polars=False)
        self.assertEqual(bridge, BRIDGE_ARROW_PANDAS)
        self.assertIsInstance(pdf, pd.DataFrame)
        self.assertEqual(list(pdf["x"]), [1, 2])

    def test_write_parquet_via_polars(self) -> None:
        import os
        import tempfile

        import pandas as pd
        import pyarrow.parquet as pq

        from chain_replay_ml.frame_backend import (
            BRIDGE_WRITE_POLARS,
            write_parquet_via_polars,
        )

        df = pd.DataFrame(
            {
                "trading_day": ["2026-01-02", "2026-01-02"],
                "token": ["A", "B"],
                "ltp": [1.0, None],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.parquet")
            bridge = write_parquet_via_polars(df, path)
            self.assertEqual(bridge, BRIDGE_WRITE_POLARS)
            table = pq.read_table(path)
            self.assertEqual(table.num_rows, 2)
            self.assertIn("ltp", table.schema.names)


if __name__ == "__main__":
    unittest.main()
