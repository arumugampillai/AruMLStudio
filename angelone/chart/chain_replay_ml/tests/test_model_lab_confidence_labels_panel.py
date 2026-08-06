"""Smoke tests for Research Lab Confidence Labels tab wiring."""

from __future__ import annotations

import inspect
import unittest


class ConfidenceLabelsTabWiringTests(unittest.TestCase):
    def test_panel_exposes_lab_refresh(self) -> None:
        from master_dataset_tk.model_lab_confidence_labels_panel import (
            ModelLabConfidenceLabelsPanel,
        )

        self.assertTrue(hasattr(ModelLabConfidenceLabelsPanel, "refresh_for_lab"))

    def test_model_lab_window_wires_confidence_subtabs(self) -> None:
        from master_dataset_tk.model_lab_window import ModelLabWindow

        self.assertTrue(hasattr(ModelLabWindow, "_build_confidence_tab"))
        self.assertTrue(hasattr(ModelLabWindow, "_build_confidence_models_pane"))
        self.assertTrue(hasattr(ModelLabWindow, "_refresh_confidence_tab"))
        src = inspect.getsource(ModelLabWindow._build_confidence_tab)
        self.assertIn("Labels", src)
        self.assertIn("Models", src)
        self.assertIn("ModelLabConfidenceLabelsPanel", src)


if __name__ == "__main__":
    unittest.main()
