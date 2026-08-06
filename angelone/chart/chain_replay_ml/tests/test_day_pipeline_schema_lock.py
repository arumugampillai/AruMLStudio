"""Schema lock + transformation summary helpers for day-at-a-time builds."""

from __future__ import annotations

import pyarrow as pa

from chain_replay_ml.dataset_builder.transformations.day_pipeline_support import (
    SchemaMismatchError,
    build_transformation_summary,
    format_transformation_summary_text,
    serialize_schema,
    validate_table_against_locked_schema,
)


def test_schema_lock_detects_missing_and_extra() -> None:
    schema = pa.schema([("a", pa.float64()), ("b", pa.float64())])
    locked = serialize_schema(schema)
    bad = pa.table({"a": [1.0], "c": [2.0]})
    try:
        validate_table_against_locked_schema(bad, locked, day="2026-07-23")
        raise AssertionError("expected SchemaMismatchError")
    except SchemaMismatchError as exc:
        assert "b" in exc.missing
        assert "c" in exc.extra


def test_schema_lock_detects_order_mismatch() -> None:
    locked = serialize_schema(pa.schema([("a", pa.float64()), ("b", pa.float64())]))
    swapped = pa.table({"b": [1.0], "a": [2.0]})
    try:
        validate_table_against_locked_schema(swapped, locked, day="Day6")
        raise AssertionError("expected order mismatch")
    except SchemaMismatchError as exc:
        assert exc.order_mismatch


def test_transformation_summary_counts_fast_safe() -> None:
    summary = build_transformation_summary(
        days=["d1", "d2", "d3"],
        mode_by_day={"d1": "fast", "d2": "safe", "d3": "fast"},
        day_stats={
            "d1": {"rows": 100, "total_sec": 10.0, "mode": "fast"},
            "d2": {"rows": 200, "total_sec": 20.0, "mode": "safe"},
            "d3": {"rows": 300, "total_sec": 15.0, "mode": "fast"},
        },
        created_columns=["x", "y"],
        feature_count=50,
        output_columns=52,
        elapsed_sec=45.0,
        codec="zstd",
        peak_ram_bytes=int(2.1 * 1024**3),
    )
    assert summary["fast_days"] == 2
    assert summary["safe_days"] == 1
    assert summary["rows"] == 600
    text = format_transformation_summary_text(summary)
    assert "Fast Days          2" in text
    assert "Safe Days          1" in text
    assert "Peak RAM           2.1 GB" in text
