"""Unit tests for benchmark stubs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from feature_intelligence.core.benchmark import (
    measure_db_latency,
    measure_memory,
    measure_time,
)


class TestBenchmark(unittest.TestCase):
    def test_measure_time_and_memory(self) -> None:
        value, timed = measure_time(lambda: 42, label="answer")
        self.assertEqual(value, 42)
        self.assertGreaterEqual(timed.elapsed_seconds, 0.0)

        value2, mem = measure_memory(lambda: list(range(1000)), label="alloc")
        self.assertEqual(len(value2), 1000)
        self.assertIsNotNone(mem.peak_memory_bytes)

    def test_db_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = measure_db_latency(Path(tmp) / "t.db", iterations=3)
            self.assertGreaterEqual(result.elapsed_seconds, 0.0)
            self.assertEqual(result.extra["iterations"], 3)


if __name__ == "__main__":
    unittest.main()
