"""Tests for master build progress adapter."""

from __future__ import annotations

import unittest


class MasterProgressAdapterTests(unittest.TestCase):
    def test_enrich_preserves_timer_pipeline(self) -> None:
        import sys
        from pathlib import Path

        chart_root = Path(__file__).resolve().parents[2]
        if str(chart_root) not in sys.path:
            sys.path.insert(0, str(chart_root))
        from master_dataset_tk.progress_adapter import enrich_master_build_payload

        raw = {
            "status": "running",
            "stage": 6,
            "message": "features: iv_zscore",
            "pipeline": {
                "stages": [
                    {"id": 1, "name": "Load Database", "status": "done", "elapsed_sec": 12.5, "elapsed_label": "12.50 s"},
                    {"id": 6, "name": "Feature Generation", "status": "running", "elapsed_sec": 40.0, "elapsed_label": "40.00 s",
                     "progress_current": 3, "progress_total": 12, "progress_unit": "groups"},
                ],
                "substages": [
                    {"id": "iv_zscore", "label": "IV Z-Score", "status": "running", "elapsed_sec": 5.0, "elapsed_label": "5.00 s"},
                ],
                "total_elapsed_sec": 52.5,
                "total_elapsed_label": "52.50 s",
                "rows_per_sec": 1200,
            },
        }
        out = enrich_master_build_payload(raw)
        pl = out["pipeline"]
        self.assertEqual(pl["substages"][0]["status"], "done")
        self.assertEqual(pl["substages"][5]["status"], "running")
        self.assertEqual(pl["substages"][5]["progress_current"], 3)
        self.assertEqual(pl.get("rows_per_sec"), 1200)
        fg = pl["substages"][-1]
        self.assertEqual(fg["id"], "iv_zscore")
        self.assertEqual(fg["parent_stage"], 6)


class MasterProgressPanelTests(unittest.TestCase):
    def test_string_substage_id_does_not_crash(self) -> None:
        import sys
        from pathlib import Path

        chart_root = Path(__file__).resolve().parents[2]
        if str(chart_root) not in sys.path:
            sys.path.insert(0, str(chart_root))
        from master_dataset_tk.progress_panel import (
            effective_substage_status,
            live_pipeline_payload,
            _substage_progress_text,
        )

        payload = {
            "status": "running",
            "stage": 6,
            "substage": "iv_zscore",
            "sub_current": 2,
            "sub_total": 12,
            "pipeline": {
                "substages": [
                    {"id": 6, "label": "Feature Generation", "status": "running", "parent_stage": "build"},
                    {"id": "iv_zscore", "label": "IV Z-Score", "status": "running", "parent_stage": 6, "elapsed_sec": 1.0},
                ],
                "total_elapsed_sec": 10.0,
            },
        }
        fg = payload["pipeline"]["substages"][1]
        self.assertEqual(effective_substage_status(fg, payload), "running")
        self.assertIn("rows", _substage_progress_text(fg, payload, "running"))
        live = live_pipeline_payload(payload, received_at=__import__("time").time() - 2)
        self.assertGreater(live["pipeline"]["total_elapsed_sec"], 10.0)


class DebugLoadProgressTests(unittest.TestCase):
    def test_debug_load_running_and_done_payloads(self) -> None:
        import sys
        from pathlib import Path

        chart_root = Path(__file__).resolve().parents[2]
        if str(chart_root) not in sys.path:
            sys.path.insert(0, str(chart_root))
        from master_dataset_tk.progress_adapter import debug_load_done_payload, debug_load_running_payload

        running = debug_load_running_payload(
            message="Loading tick DB",
            current=1,
            total=2,
            source_day_index=1,
            source_day_total=2,
            ticks_in_memory=500_000,
            spot_ticks=100_000,
            chain_ticks=400_000,
            elapsed_sec=12.5,
        )
        self.assertTrue(running.get("debug_load"))
        self.assertEqual(running["status"], "running")
        self.assertEqual(running["ticks_in_memory"], 500_000)
        self.assertEqual(running["pipeline"]["substages"][0]["status"], "running")

        done = debug_load_done_payload(
            ticks_in_memory=1_200_000,
            spot_ticks=200_000,
            chain_ticks=1_000_000,
            sources_loaded=2,
            elapsed_sec=45.0,
        )
        self.assertEqual(done["status"], "completed")
        self.assertIn("1,200,000", done["message"])
        self.assertEqual(done["pipeline"]["stages"][0]["status"], "done")
        self.assertEqual(done["pipeline"]["substages"][0]["status"], "done")
        for sub in done["pipeline"]["substages"][1:]:
            self.assertEqual(sub["status"], "waiting")

    def test_debug_feature_done_payload(self) -> None:
        import sys
        from pathlib import Path

        chart_root = Path(__file__).resolve().parents[2]
        if str(chart_root) not in sys.path:
            sys.path.insert(0, str(chart_root))
        from master_dataset_tk.progress_adapter import debug_feature_done_payload

        pl = {
            "stages": [
                {"id": 1, "name": "Load Database", "status": "done", "elapsed_sec": 24.0, "elapsed_label": "24.00 s"},
                {"id": 6, "name": "Feature Generation", "status": "done", "elapsed_sec": 96.0, "elapsed_label": "96.00 s"},
            ],
            "substages": [],
            "total_elapsed_sec": 120.5,
            "total_elapsed_label": "120.50 s",
        }
        done = debug_feature_done_payload(
            rows=310_002,
            feature_count=200,
            groups_run=20,
            sources_loaded=1,
            elapsed_sec=120.5,
            ticks_in_memory=2_668_802,
            spot_ticks=91_126,
            chain_ticks=2_577_676,
            pipeline=pl,
        )
        self.assertEqual(done["status"], "completed")
        self.assertTrue(done.get("debug_features"))
        self.assertEqual(done["rows"], 310_002)
        self.assertEqual(done["feature_count"], 200)
    def test_feature_group_progress_text(self) -> None:
        import sys
        from pathlib import Path

        chart_root = Path(__file__).resolve().parents[2]
        if str(chart_root) not in sys.path:
            sys.path.insert(0, str(chart_root))
        from master_dataset_tk.progress_panel import _substage_progress_text

        payload = {
            "stage": 6,
            "feature_groups_done": 3,
            "feature_groups_total": 20,
            "feature_groups_remaining": 17,
            "feature_group_id": "momentum",
            "feature_group_current": "Momentum",
            "sub_current": 50_000,
            "sub_total": 310_002,
        }
        stage6 = {"id": 6, "label": "Feature Generation", "status": "running"}
        self.assertEqual(_substage_progress_text(stage6, payload, "running"), "3 / 20 groups")
        fg = {"id": "momentum", "label": "Momentum", "status": "running"}
        self.assertIn("rows", _substage_progress_text(fg, payload, "running"))

    def test_build_progress_emit_feature_groups(self) -> None:
        from chain_replay_ml.dataset_builder.progress import (
            BuildProgress,
            feature_group_progress_fields,
        )

        progress = BuildProgress("job-1")
        payload = progress.emit(
            stage=6,
            **feature_group_progress_fields(
                groups_done=3,
                groups_total=20,
                group_id="momentum",
                group_label="Momentum",
                row_current=100,
                row_total=500,
            ),
        )
        self.assertEqual(payload["feature_groups_done"], 3)
        self.assertEqual(payload["feature_groups_total"], 20)
        self.assertEqual(payload["feature_groups_remaining"], 17)
        self.assertEqual(payload["feature_group_current"], "Momentum")


if __name__ == "__main__":
    unittest.main()
