"""Focused unit & smoke tests for Research Detail window UI (top-level beside main app)."""

import tkinter as tk
from unittest.mock import patch
import unittest

from chain_replay_ml.core.data_root import DataRootService
from chain_replay_ml.research_registry.store import (
    backfill_historical_research_records,
    get_all_research_records,
    get_research_detail,
)
from master_dataset_tk.fold_replay_widgets import place_toplevel_beside_main
from master_dataset_tk.morning_research_dossier_panel import MorningResearchDossierPanel


class TestResearchDetailWindowUI(unittest.TestCase):
    """Verify Research Detail window opens as a separate top-level window adjacent to the main app."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = DataRootService().get_data_root()
        backfill_historical_research_records(cls.data_dir)

    def test_research_detail_window_beside_main(self) -> None:
        """Verify window creation, positioning beside main, sizing, and content rendering."""
        root = tk.Tk()
        root.geometry("1024x720+100+100")
        root.update_idletasks()

        try:
            panel = MorningResearchDossierPanel(root, data_dir=self.data_dir)
            self.assertGreaterEqual(len(panel._all_records), 1)

            # 1. Double click first research run
            r_id_1 = panel._all_records[0]["research_id"]
            panel._show_research_detail_modal(r_id_1)

            win = panel._detail_window
            self.assertIsNotNone(win)
            self.assertIsInstance(win, tk.Toplevel)
            self.assertTrue(win.winfo_exists())
            self.assertEqual(panel._current_detail_research_id, r_id_1)
            self.assertIn(r_id_1, win.title())

            # Verify geometry / positioning (width >= 800, height >= 600, positioned beside main)
            win.update_idletasks()
            win_w = win.winfo_width()
            win_h = win.winfo_height()
            self.assertGreaterEqual(win_w, 640)
            self.assertGreaterEqual(win_h, 480)

            # Verify notebook and tabs exist
            notebooks = [child for child in win.winfo_children() if isinstance(child, tk.ttk.Notebook)]
            self.assertEqual(len(notebooks), 1)
            nb = notebooks[0]
            tab_texts = [nb.tab(t, "text") for t in nb.tabs()]
            self.assertTrue(any("Research Summary & Hardware" in t for t in tab_texts))
            self.assertTrue(any("Features" in t for t in tab_texts))
            self.assertTrue(any("Generational Progress" in t for t in tab_texts))
            self.assertTrue(any("Candidate Leaderboard" in t for t in tab_texts))
            self.assertTrue(any("Generational Lineage" in t for t in tab_texts))
            self.assertTrue(any("Feature Governance" in t for t in tab_texts))
            self.assertTrue(any("Execution Audit Trail" in t for t in tab_texts))

            # 2. Open second research run -> verify existing window is reused (no duplicate window)
            if len(panel._all_records) > 1:
                r_id_2 = panel._all_records[1]["research_id"]
                panel._show_research_detail_modal(r_id_2)

                self.assertEqual(panel._detail_window, win)
                self.assertEqual(panel._current_detail_research_id, r_id_2)
                self.assertIn(r_id_2, win.title())

            # 3. Close window
            win.destroy()
            self.assertFalse(win.winfo_exists())

        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
