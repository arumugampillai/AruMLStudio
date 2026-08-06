"""Research Lab Triple Barrier model dropdown — discovery filter."""

from __future__ import annotations

import unittest

from master_dataset_tk.model_lab_window import tb_model_names_from_registry_rows


class TbModelComboDiscoveryTests(unittest.TestCase):
    def test_finds_tb_prefixed_label_id_model(self) -> None:
        rows = [
            {
                "model_name": "TB_tp_20_sl_10_WF_395f_XGB_1234_1",
                "target": "label_id",
                "label_strategy": "triple_barrier",
            }
        ]
        self.assertEqual(
            tb_model_names_from_registry_rows(rows),
            ["TB_tp_20_sl_10_WF_395f_XGB_1234_1"],
        )

    def test_finds_model_by_label_id_target_without_tb_prefix(self) -> None:
        """Older/renamed TB packages may lack the ``TB_`` prefix — target still identifies them."""
        rows = [{"model_name": "Custom_Renamed_Model", "target": "label_id"}]
        self.assertEqual(
            tb_model_names_from_registry_rows(rows),
            ["Custom_Renamed_Model"],
        )

    def test_finds_model_by_legacy_label_strategy_id_field(self) -> None:
        rows = [
            {
                "model_name": "Legacy_TB_Model",
                "label_strategy_id": "triple_barrier",
            }
        ]
        self.assertEqual(
            tb_model_names_from_registry_rows(rows),
            ["Legacy_TB_Model"],
        )

    def test_excludes_regression_models(self) -> None:
        rows = [
            {
                "model_name": "Future_LTP_5m_TSS_100f_XGB_0900_1",
                "target": "future_ltp_5m",
                "label_strategy": "fixed_horizon",
            }
        ]
        self.assertEqual(tb_model_names_from_registry_rows(rows), [])

    def test_mixed_rows_preserve_input_order_newest_first(self) -> None:
        """Input rows are expected newest-first (per the global Model sort
        standard) — this function must not re-sort them alphabetically."""
        rows = [
            {"model_name": "Zeta_TB", "target": "label_id"},
            {"model_name": "Future_LTP_5m_TSS_100f_XGB_0900_1", "target": "future_ltp_5m"},
            {"model_name": "Alpha_TB", "label_strategy": "triple_barrier"},
        ]
        self.assertEqual(
            tb_model_names_from_registry_rows(rows),
            ["Zeta_TB", "Alpha_TB"],
        )

    def test_dedupes_while_preserving_order(self) -> None:
        rows = [
            {"model_name": "Zeta_TB", "target": "label_id"},
            {"model_name": "Alpha_TB", "label_strategy": "triple_barrier"},
            {"model_name": "Zeta_TB", "target": "label_id"},
        ]
        self.assertEqual(
            tb_model_names_from_registry_rows(rows),
            ["Zeta_TB", "Alpha_TB"],
        )

    def test_ignores_rows_missing_model_name(self) -> None:
        rows = [{"target": "label_id"}]
        self.assertEqual(tb_model_names_from_registry_rows(rows), [])

    def test_empty_input(self) -> None:
        self.assertEqual(tb_model_names_from_registry_rows([]), [])


if __name__ == "__main__":
    unittest.main()
