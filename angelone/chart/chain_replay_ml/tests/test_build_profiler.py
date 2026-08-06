"""Tests for dataset build profiler aggregation."""

from __future__ import annotations

from chain_replay_ml.dataset_builder.build_profiler import (
    BuildProfiler,
    profile_block,
    set_profiler,
)


def test_build_profiler_records_and_sorts() -> None:
    prof = BuildProfiler()
    prof.start_build()
    set_profiler(prof)
    try:
        with profile_block("stage.load_database", rows=100):
            with profile_block("function.enrich_dataset_features", rows=1):
                pass
        with profile_block("stage.sqlite_insert", rows=100):
            pass
        prof.record("controller.dgt", 2.5, rows=100)
        prof.record("controller.dgt", 1.5, rows=100)
    finally:
        set_profiler(None)
    prof.finish_build(total_rows=100)
    report = prof.to_report()
    assert report["total_rows"] == 100
    assert report["stages"]
    assert report["functions"]
    assert report["controllers"]
    ranked = report["ranked"]
    assert ranked[0]["total_sec"] >= ranked[-1]["total_sec"]
    dgt = next(e for e in report["controllers"] if e["name"] == "controller.dgt")
    assert dgt["call_count"] == 2
    assert dgt["rows"] == 200


def test_profile_block_noop_when_inactive() -> None:
    set_profiler(None)
    with profile_block("stage.load_database"):
        pass
