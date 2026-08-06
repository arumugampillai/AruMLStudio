"""Unit tests for master-build Stage-6 one-pass + parallel mode defaults."""

from __future__ import annotations

from chain_replay_ml.dataset_builder.production_day_build import _default_parallel_mode
from chain_replay_ml.dataset_builder.stages_parallel import _group_rows_by_token


def test_default_parallel_mode_is_token(monkeypatch) -> None:
    monkeypatch.delenv("MASTER_BUILD_PARALLEL", raising=False)
    assert _default_parallel_mode() == "token"
    monkeypatch.setenv("MASTER_BUILD_PARALLEL", "serial")
    assert _default_parallel_mode() == "serial"
    monkeypatch.setenv("MASTER_BUILD_PARALLEL", "off")
    assert _default_parallel_mode() == "serial"


def test_group_rows_by_token_preserves_locality() -> None:
    rows = [
        {"token": "CE50", "timestamp": 20.0, "strike": 50, "option_type": "CE", "_atm": 100},
        {"token": "CE50", "timestamp": 10.0, "strike": 50, "option_type": "CE", "_atm": 100},
        {"token": "PE50", "timestamp": 15.0, "strike": 50, "option_type": "PE", "_atm": 100},
    ]
    grouped = _group_rows_by_token(rows)
    assert list(grouped.keys()) == ["CE50", "PE50"] or set(grouped) == {"CE50", "PE50"}
    assert [s["timestamp"] for _, s in grouped["CE50"]] == [10.0, 20.0]
