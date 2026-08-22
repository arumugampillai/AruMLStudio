"""Focused smoke test for Autonomous Research Detail Dossier."""

import json
import tkinter as tk
import unittest

from chain_replay_ml.core.data_root import DataRootService
from chain_replay_ml.research_registry.store import (
    backfill_historical_research_records,
    get_all_research_records,
    get_research_detail,
)
from master_dataset_tk.morning_research_dossier_panel import MorningResearchDossierPanel


class TestResearchDetailDossierSmoke(unittest.TestCase):
    """Smoke test for Autonomous Research Detail Dossier dialog and data store."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = DataRootService().get_data_root()
        backfill_historical_research_records(cls.data_dir)

    def test_get_research_detail_data_structure(self) -> None:
        """Verify get_research_detail returns complete historical telemetry."""
        records = get_all_research_records(self.data_dir)
        self.assertGreaterEqual(len(records), 1, "Expected at least 1 research record")

        # Pick a research run with features if possible
        target_rec = next((r for r in records if r.get("total_df_features_created", 0) > 0), records[0])
        r_id = target_rec["research_id"]

        detail = get_research_detail(self.data_dir, r_id)
        self.assertIsNotNone(detail)
        self.assertEqual(detail["research_id"], r_id)

        # 1. Verify summary fields
        self.assertIn("campaign_id", detail)
        self.assertIn("context_key", detail)
        self.assertIn("dataset_name", detail)
        self.assertIn("status", detail)
        self.assertIn("best_candidate_id", detail)
        self.assertIn("best_composite_score", detail)

        # 2. Verify hardware metadata
        hw = detail.get("hardware", {})
        self.assertIn("cpu", hw)
        self.assertIn("gpu", hw)
        self.assertIn("gpu_model", hw)
        self.assertIn("algorithms_mapping", hw)

        # 3. Verify features list & verdicts
        features = detail.get("features", [])
        if target_rec.get("total_df_features_created", 0) > 0:
            self.assertGreater(len(features), 0)
            f0 = features[0]
            self.assertIn("feature_name", f0)
            self.assertIn("feature_id", f0)
            self.assertIn("formula_hash", f0)
            self.assertIn("lifecycle_status", f0)
            self.assertIn("evidence_score", f0)

        # 4. Verify generations progression
        gens = detail.get("generations", [])
        self.assertIsInstance(gens, list)

    def test_research_detail_modal_ui(self) -> None:
        """Instantiate MorningResearchDossierPanel and open research detail modal dialog."""
        try:
            root = tk.Tk()
            root.withdraw()
        except tk.TclError:
            self.skipTest("Headless environment without display server")

        try:
            panel = MorningResearchDossierPanel(root, data_dir=self.data_dir)
            records = get_all_research_records(self.data_dir)
            self.assertGreater(len(records), 0)

            target_rec = next((r for r in records if r.get("total_df_features_created", 0) > 0), records[0])
            r_id = target_rec["research_id"]

            # Open modal
            panel._show_research_detail_modal(r_id)

            # Find the opened Toplevel
            toplevels = [w for w in panel.winfo_children() if isinstance(w, tk.Toplevel)] + [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
            self.assertGreater(len(toplevels), 0, "Modal dialog not opened")
            top = toplevels[0]

            # Verify Notebook inside Toplevel
            notebooks = [w for w in top.winfo_children() if isinstance(w, tk.ttk.Notebook)]
            self.assertEqual(len(notebooks), 1)
            nb = notebooks[0]

            tab_texts = [nb.tab(t, "text") for t in nb.tabs()]
            self.assertTrue(any("Research Summary & Hardware" in t for t in tab_texts))
            self.assertTrue(any("Features" in t for t in tab_texts))
            self.assertTrue(any("Generational Progress" in t for t in tab_texts))

            top.destroy()
            root.destroy()
        except Exception:
            root.destroy()
            raise


if __name__ == "__main__":
    unittest.main()
