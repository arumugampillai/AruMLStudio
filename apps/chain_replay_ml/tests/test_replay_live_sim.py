"""Verify live sim passes replay expiry into scored-frame build."""
from __future__ import annotations

import pandas as pd


def test_live_sim_passes_expiry_hint(monkeypatch):
    captured: list[dict] = []

    def fake_build(data_dir, model_name, date_str, **kwargs):
        captured.append(kwargs)
        return pd.DataFrame(
            {
                "timestamp": [1000.0],
                "token": ["1"],
                "symbol": ["NIFTY07JUL2624500CE"],
                "ltp": [100.0],
                "score": [5.0],
                "P_hit": [0.6],
                "delta_band": ["0.35-0.45"],
                "spot": [24500.0],
                "strike": [24500.0],
                "option_type": ["CE"],
            }
        )

    class TL:
        def ltp_at(self, ts):
            return 100.0

    monkeypatch.setattr(
        "chain_replay_ml.registry_backtest.build_registry_scored_frame",
        fake_build,
    )
    monkeypatch.setattr(
        "chain_replay_ml.training.default_model.resolve_default_model_name",
        lambda data_dir, model_name: model_name,
    )
    monkeypatch.setattr(
        "chain_replay_ml.replay_live_sim.load_tick_timelines",
        lambda conn, tokens, open_ts, close_ts: {"1": TL()},
    )
    monkeypatch.setattr(
        "chain_replay_ml.replay_live_sim.check_scalp_outcome_seconds_config_b",
        lambda tl, ts, window, tgt, sl: (0, 60.0, 101.0, ts + 60.0),
    )
    monkeypatch.setattr(
        "chain_replay_ml.replay_live_sim.replay_db_path",
        lambda chart_dir, date_str: ":memory:",
    )

    from chain_replay_ml.replay_live_sim import run_replay_live_sim

    result = run_replay_live_sim(
        date_str="2026-07-02",
        expiry="2026-07-07",
        model_name="Test_Model",
        position_limit=1,
    )

    assert captured == [{"expiry_hint": "2026-07-07"}]
    assert result["signal_count"] == 1


def test_live_sim_position_limit_blocks_overlap(monkeypatch):
    """Concurrent position cap matches report simulate_positions (exit_ts based)."""

    def fake_build(data_dir, model_name, date_str, **kwargs):
        return pd.DataFrame(
            {
                "timestamp": [1000.0, 1010.0],
                "token": ["1", "1"],
                "symbol": ["NIFTY07JUL2624500CE", "NIFTY07JUL2624500CE"],
                "ltp": [100.0, 100.0],
                "score": [5.0, 5.0],
                "P_hit": [0.6, 0.6],
                "delta_band": ["0.35-0.45", "0.35-0.45"],
                "spot": [24500.0, 24500.0],
                "strike": [24500.0, 24500.0],
                "option_type": ["CE", "CE"],
            }
        )

    class TL:
        def ltp_at(self, ts):
            return 100.0

    monkeypatch.setattr(
        "chain_replay_ml.registry_backtest.build_registry_scored_frame",
        fake_build,
    )
    monkeypatch.setattr(
        "chain_replay_ml.training.default_model.resolve_default_model_name",
        lambda data_dir, model_name: model_name,
    )
    monkeypatch.setattr(
        "chain_replay_ml.replay_live_sim.load_tick_timelines",
        lambda conn, tokens, open_ts, close_ts: {"1": TL()},
    )
    monkeypatch.setattr(
        "chain_replay_ml.replay_live_sim.check_scalp_outcome_seconds_config_b",
        lambda tl, ts, window, tgt, sl: (0, 300.0, 101.0, ts + 300.0),
    )
    monkeypatch.setattr(
        "chain_replay_ml.replay_live_sim.replay_db_path",
        lambda chart_dir, date_str: ":memory:",
    )

    from chain_replay_ml.replay_live_sim import run_replay_live_sim

    result = run_replay_live_sim(
        date_str="2026-07-02",
        expiry="2026-07-07",
        model_name="Test_Model",
        position_limit=1,
    )

    assert result["entered_count"] == 1
    skips = [s for s in result["signals"] if s["action"] == "SKIP"]
    assert any("Position limit" in s.get("reason", "") for s in skips)
