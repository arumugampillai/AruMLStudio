"""Tests for Phase 6 research intelligence modules."""

from __future__ import annotations

import tempfile
import unittest

from chain_replay_ml.fold_research.csv_export import build_fold_research_csv
from chain_replay_ml.fold_research.error_explorer import rank_prediction_errors
from chain_replay_ml.fold_research.notes_store import ResearchNotesStore, save_fold_note, search_research_notes
from chain_replay_ml.fold_research.regime_detection import analyze_regimes


class ResearchIntelligenceTests(unittest.TestCase):
    def test_rank_prediction_errors(self) -> None:
        rows = [
            {"prediction_id": "a", "prediction_error": 5.0, "token": "T1"},
            {"prediction_id": "b", "prediction_error": -8.0, "token": "T2"},
            {"prediction_id": "c", "prediction_error": 2.0, "token": "T3"},
        ]
        abs_rank = rank_prediction_errors(rows, mode="absolute", limit=2)
        self.assertEqual(abs_rank[0]["prediction_id"], "b")
        pos = rank_prediction_errors(rows, mode="positive", limit=1)
        self.assertEqual(pos[0]["prediction_id"], "a")

    def test_research_notes_roundtrip(self) -> None:
        tmp = tempfile.mkdtemp()
        note = save_fold_note(
            tmp,
            prediction_run_id="run1",
            fold_id="fold1",
            title="IV day",
            body="High IV expansion after 2 PM.",
            tags=["IV expansion"],
        )
        self.assertTrue(note.get("note_id"))
        hits = search_research_notes(tmp, "IV expansion")
        self.assertEqual(len(hits), 1)
        with ResearchNotesStore(tmp) as store:
            listed = store.list_notes(prediction_run_id="run1", fold_id="fold1")
        self.assertEqual(len(listed), 1)

    def test_csv_export_sections(self) -> None:
        detail = {
            "ok": True,
            "fold": {"fold_number": 5, "fold_id": "f5", "validation_rows": 1000},
            "prediction_run": {"model_id": "M1", "run_id": "r1"},
            "prediction_quality": {
                "row_count": 1000,
                "mae": 3.17,
                "calibration_buckets": [{"bin": 1, "count": 200, "pred_return_avg_pct": 1.0, "actual_return_avg_pct": 0.8, "calibration_error_pct": -0.2}],
            },
            "market_summary": {"trading_days": ["2026-05-27"], "spot_start": 1, "spot_end": 2},
            "trading": {"metrics": {"trade_count": 13, "profit": 100}, "trades": []},
            "error_explorer": {"absolute": rank_prediction_errors([{"prediction_error": 1.0, "prediction_id": "p1"}], limit=10), "positive": [], "negative": []},
            "feature_drift": {"top_drifted": [{"feature": "ltp", "train_mean": 1.0, "validation_mean": 1.1, "shift_pct": 10, "severity": "High"}]},
            "regime_analysis": {"regimes": [{"regime": "Range", "row_count": 100, "mae": 1.2}]},
        }
        csv_text = build_fold_research_csv("", detail)
        for section in ("Overview", "Prediction", "Trading", "Error Explorer", "Feature Drift", "Regime"):
            self.assertIn(section, csv_text)

    def test_regime_analysis(self) -> None:
        rows = [
            {"timestamp": 1.0, "spot": 100.0, "ltp": 20.0, "prediction_error": 1.0},
            {"timestamp": 2.0, "spot": 100.5, "ltp": 21.0, "prediction_error": 2.0},
        ]
        doc = analyze_regimes(rows)
        self.assertTrue(doc.get("available"))
        self.assertGreaterEqual(len(doc.get("regimes") or []), 1)

    def test_trade_replay_timeline(self) -> None:
        from chain_replay_ml.fold_research.trade_replay import build_trade_replay

        trade = {
            "trade_id": "t1",
            "token": "TOK_A",
            "entry_ts": 100.0,
            "exit_ts": 109.0,
            "entry_price": 20.0,
            "exit_price": 23.0,
            "net_pnl": 100.0,
            "exit_reason": "target",
            "entry_prediction_id": "r:f:0",
        }
        rows = [
            {"prediction_id": "r:f:0", "token": "TOK_A", "timestamp": 100.0, "spot": 24000, "ltp": 20.0, "predicted_ltp": 22.0, "row_index": 0},
            {"prediction_id": "r:f:1", "token": "TOK_A", "timestamp": 103.0, "ltp": 21.0, "actual_ltp": 21.0, "row_index": 1},
            {"prediction_id": "r:f:2", "token": "TOK_A", "timestamp": 106.0, "ltp": 23.0, "actual_ltp": 23.0, "row_index": 2},
            {"prediction_id": "r:f:3", "token": "TOK_A", "timestamp": 109.0, "ltp": 23.0, "actual_ltp": 23.0, "row_index": 3},
        ]
        doc = build_trade_replay(trade, rows, cfg={"entry": {"direction": "long", "premium_min": 10, "premium_max": 50}})
        self.assertTrue(doc.get("ok"))
        types = {e["event_type"] for e in doc["events"]}
        self.assertIn("strategy_entry", types)
        self.assertIn("premium_tick", types)
        self.assertIn("trade_exit", types)
        self.assertIn("prediction", doc.get("decision", {}))
        self.assertIn("exit_analysis", doc)
        self.assertIn("price_paths", doc)
        self.assertIn("trade_verdict", doc)
        self.assertIn("maximum_opportunity", doc)
        self.assertIn("since_entry", doc)
        self.assertIn("pnl_path", doc)
        self.assertIn("rule_timeline", doc)
        self.assertIn("feature_alerts", doc)
        self.assertIn("research_conclusion", doc)
        self.assertIn("trade_classification", doc)
        self.assertIn("prediction_failure", doc)
        self.assertIn("regime_badges", doc)
        self.assertIn("model", doc.get("trade_verdict") or {})
        self.assertIn("strategy", doc.get("trade_verdict") or {})
        self.assertIn("decision_quality", doc.get("decision", {}))
        doc2 = build_trade_replay(
            trade,
            rows,
            cfg={"entry": {"direction": "long", "premium_min": 10, "premium_max": 50}},
            peer_trades=[trade, {"trade_id": "t2", "entry_price": 21.0, "net_pnl": -50, "exit_reason": "stop", "holding_seconds": 30}],
        )
        self.assertGreaterEqual(len(doc2.get("similar_trades") or []), 1)
        self.assertIn("matched_on", (doc2.get("similar_trades") or [{}])[0])

    def test_generate_trade_observation(self) -> None:
        from chain_replay_ml.fold_research.trade_replay import build_trade_replay, generate_trade_observation

        trade = {
            "trade_id": "t1",
            "token": "TOK_A",
            "entry_ts": 100.0,
            "exit_ts": 109.0,
            "entry_price": 20.0,
            "exit_price": 18.0,
            "net_pnl": -50.0,
            "exit_reason": "stop",
            "entry_prediction_id": "r:f:0",
            "holding_seconds": 9,
        }
        rows = [
            {
                "prediction_id": "r:f:0",
                "token": "TOK_A",
                "timestamp": 100.0,
                "spot": 24000,
                "ltp": 20.0,
                "predicted_ltp": 25.0,
                "actual_ltp": 18.0,
                "direction_correct": 0,
                "row_index": 0,
            },
            {"prediction_id": "r:f:1", "token": "TOK_A", "timestamp": 106.0, "ltp": 18.0, "actual_ltp": 18.0, "row_index": 1},
            {"prediction_id": "r:f:2", "token": "TOK_A", "timestamp": 109.0, "ltp": 18.0, "actual_ltp": 18.0, "row_index": 2},
        ]
        doc = build_trade_replay(trade, rows, cfg={"entry": {"direction": "long", "premium_min": 10, "premium_max": 50}})
        obs = generate_trade_observation(doc)
        self.assertIn("lost because", obs.get("body", "").lower())
        self.assertGreaterEqual(len(obs.get("bullets") or []), 1)
        self.assertTrue(obs.get("tags"))

    def test_counterfactual_replay(self) -> None:
        from chain_replay_ml.fold_research.counterfactual import build_counterfactuals

        trade = {"entry_price": 20.0, "entry_ts": 100.0, "net_pnl": -2.0, "exit_reason": "stop", "holding_seconds": 9}
        path = [
            {"timestamp": 100.0, "value": 20.0},
            {"timestamp": 103.0, "value": 21.5},
            {"timestamp": 106.0, "value": 18.0},
            {"timestamp": 109.0, "value": 18.5},
        ]
        doc = build_counterfactuals(trade, path, cfg={"stop": {"stop_loss_pct": 5}, "target": {"target_profit_pct": 8}, "hold_time": {"max_hold_sec": 30}})
        self.assertTrue(doc.get("available"))
        self.assertGreaterEqual(len(doc.get("scenarios") or []), 4)

    def test_fold_quality(self) -> None:
        from chain_replay_ml.fold_research.fold_quality import compute_fold_quality

        fq = compute_fold_quality(
            prediction_quality={"mae": 1.2, "directional_accuracy_pct": 62},
            trading_metrics={"profit": 500, "win_rate_pct": 55, "profit_factor": 1.4, "max_drawdown": -200},
            regime_analysis={"regimes": [{"row_count": 40}, {"row_count": 30}]},
            trade_count=20,
        )
        self.assertGreaterEqual(fq.get("total", 0), 50)


if __name__ == "__main__":
    unittest.main()
