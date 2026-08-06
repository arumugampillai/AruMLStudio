"""Smoke tests for Research Lab Strategy Simulator tab wiring."""

from __future__ import annotations

import unittest


class StrategySimTabWiringTests(unittest.TestCase):
    def test_panel_exposes_lab_refresh(self) -> None:
        from master_dataset_tk.model_lab_strategy_sim_panel import ModelLabStrategySimPanel

        self.assertTrue(hasattr(ModelLabStrategySimPanel, "refresh_for_lab"))

    def test_model_lab_window_exposes_strategy_sim_helpers(self) -> None:
        from master_dataset_tk.model_lab_window import ModelLabWindow

        self.assertTrue(hasattr(ModelLabWindow, "_build_strategy_sim_tab"))
        self.assertTrue(hasattr(ModelLabWindow, "_refresh_strategy_sim_tab"))
        self.assertTrue(hasattr(ModelLabWindow, "select_strategy_sim_tab"))

    def test_panel_exposes_execution_rules_helpers(self) -> None:
        from master_dataset_tk.model_lab_strategy_sim_panel import ModelLabStrategySimPanel

        self.assertTrue(hasattr(ModelLabStrategySimPanel, "_build_execution_rules_tab"))
        self.assertTrue(hasattr(ModelLabStrategySimPanel, "_execution_rules_from_ui"))
        self.assertTrue(hasattr(ModelLabStrategySimPanel, "_sync_execution_rules_controls"))
        self.assertTrue(hasattr(ModelLabStrategySimPanel, "_build_worst_open_risk_tab"))
        self.assertTrue(hasattr(ModelLabStrategySimPanel, "_render_worst_open_risk"))
        self.assertTrue(hasattr(ModelLabStrategySimPanel, "_render_stop_tick_replay"))
        self.assertTrue(hasattr(ModelLabStrategySimPanel, "_fill_timeline_tree"))
        self.assertTrue(hasattr(ModelLabStrategySimPanel, "_scrollable_host"))
        self.assertTrue(hasattr(ModelLabStrategySimPanel, "_render_overview_daily_pnl_table"))
        self.assertTrue(hasattr(ModelLabStrategySimPanel, "_daily_pnl_rows"))
        self.assertTrue(hasattr(ModelLabStrategySimPanel, "_download_equity_curve_csv"))
        self.assertTrue(hasattr(ModelLabStrategySimPanel, "_download_trades_csv"))
        self.assertTrue(hasattr(ModelLabStrategySimPanel, "_build_charges_tab"))
        self.assertTrue(hasattr(ModelLabStrategySimPanel, "_render_charges"))
        self.assertTrue(hasattr(ModelLabStrategySimPanel, "_edit_selected_strategy"))
        self.assertTrue(hasattr(ModelLabStrategySimPanel, "_selected_strategy_context"))
        self.assertTrue(hasattr(ModelLabStrategySimPanel, "_load_ui_prefs"))
        self.assertTrue(hasattr(ModelLabStrategySimPanel, "_save_ui_prefs"))

    def test_panel_exposes_package_filter_helpers(self) -> None:
        from master_dataset_tk.model_lab_strategy_sim_panel import ModelLabStrategySimPanel

        for name in (
            "_build_package_filter_section",
            "_build_classifier_threshold_tab",
            "_refresh_package_filter_list",
            "_selected_package_option",
            "_on_package_filter_selected",
            "_sync_threshold_tab",
            "_render_threshold_table",
            "_on_threshold_picked",
            "_refresh_package_filter_summary",
        ):
            self.assertTrue(hasattr(ModelLabStrategySimPanel, name), name)

    def test_panel_exposes_tb_filter_helpers(self) -> None:
        from master_dataset_tk.model_lab_strategy_sim_panel import ModelLabStrategySimPanel

        for name in (
            "_build_tb_filter_section",
            "_sync_tb_filter_controls",
            "_on_tb_filter_toggled",
            "_refresh_tb_filter_options",
            "_selected_tb_class",
            "_tb_filter_kwargs_from_ui",
            "_refresh_tb_filter_summary",
            "_tb_filter_prefs",
            "_build_confidence_threshold_subtab",
            "_build_threshold_stub_subtab",
            "_render_tb_summary_section",
            "_render_tb_comparison_section",
            "_build_tb_threshold_subtab",
            "_render_tb_threshold_table",
            "_on_tb_threshold_row_picked",
        ):
            self.assertTrue(hasattr(ModelLabStrategySimPanel, name), name)

    def test_tb_filter_prefs_roundtrip(self) -> None:
        import tempfile

        from master_dataset_tk.strategy_sim_prefs import (
            load_strategy_sim_prefs,
            save_strategy_sim_prefs,
        )

        with tempfile.TemporaryDirectory() as tmp:
            save_strategy_sim_prefs(
                tmp,
                {"triple_barrier_filter": {"enabled": True, "class_label": "TP", "threshold": 0.65}},
            )
            tb = load_strategy_sim_prefs(tmp).get("triple_barrier_filter") or {}
            self.assertTrue(tb.get("enabled"))
            self.assertEqual(tb.get("class_label"), "TP")
            self.assertEqual(tb.get("threshold"), 0.65)

    def test_probability_filter_prefs_roundtrip(self) -> None:
        import tempfile

        from master_dataset_tk.strategy_sim_prefs import (
            load_strategy_sim_prefs,
            save_strategy_sim_prefs,
        )

        with tempfile.TemporaryDirectory() as tmp:
            save_strategy_sim_prefs(
                tmp,
                {"probability_filter": {"member_key": "up_3pct", "threshold": 0.7}},
            )
            prob = load_strategy_sim_prefs(tmp).get("probability_filter") or {}
            self.assertEqual(prob.get("member_key"), "up_3pct")
            self.assertEqual(prob.get("threshold"), 0.7)

    def test_strategy_sim_prefs_roundtrip(self) -> None:
        import tempfile

        from master_dataset_tk.strategy_sim_prefs import (
            load_strategy_sim_prefs,
            save_strategy_sim_prefs,
        )

        with tempfile.TemporaryDirectory() as tmp:
            save_strategy_sim_prefs(
                tmp,
                {
                    "strategy_version_id": "abc123",
                    "strategy_id": "sid1",
                    "execution_rules": {
                        "enabled": True,
                        "max_open_positions": 1,
                        "one_position_per_symbol": True,
                    },
                },
            )
            prefs = load_strategy_sim_prefs(tmp)
            self.assertEqual(prefs.get("strategy_version_id"), "abc123")
            self.assertEqual(prefs.get("strategy_id"), "sid1")
            self.assertTrue((prefs.get("execution_rules") or {}).get("enabled"))
            self.assertEqual((prefs.get("execution_rules") or {}).get("max_open_positions"), 1)

    def test_strategy_rule_rows_helper(self) -> None:
        from master_dataset_tk.model_lab_strategy_sim_panel import _strategy_rule_rows

        rows = dict(
            _strategy_rule_rows(
                {
                    "entry": {
                        "direction": "long",
                        "premium_min": 50,
                        "premium_max": 100,
                        "atm_band": 15,
                        "expiry": "current",
                        "entry_cadence_sec": 3,
                        "option_types": ["CE", "PE"],
                    },
                    "stop": {"stop_loss_pct": 4.0},
                    "target": {"target_profit_pct": 6.0},
                    "hold_time": {"max_hold_sec": 300},
                }
            )
        )
        self.assertEqual(rows["Stop loss"], "4.0%")
        self.assertEqual(rows["Target"], "6.0%")
        self.assertEqual(rows["Max hold"], "300s")
        self.assertEqual(rows["Min predicted move"], "Off (direction only)")

        rows2 = dict(
            _strategy_rule_rows(
                {
                    "entry": {"minimum_predicted_move_pct": 5.0},
                    "stop": {},
                    "target": {"use_predicted_ltp": True, "target_profit_pct": 6.0},
                    "hold_time": {},
                }
            )
        )
        self.assertEqual(rows2["Min predicted move"], "5%")
        self.assertEqual(rows2["Target"], "Predicted LTP (entry row)")

    def test_probability_filter_cell_format(self) -> None:
        from master_dataset_tk.model_lab_strategy_sim_panel import (
            _format_probability_filter_cell,
        )

        self.assertEqual(
            _format_probability_filter_cell(
                {
                    "probability_filter_active": True,
                    "probability_filter_label": "+2%",
                    "probability_filter_threshold": 0.7,
                }
            ),
            "2%(0.70)",
        )
        self.assertEqual(
            _format_probability_filter_cell(
                {
                    "probability_filter_active": True,
                    "probability_filter_label": ">6% Probability",
                    "probability_filter_threshold": 0.55,
                }
            ),
            ">6%(0.55)",
        )
        self.assertEqual(
            _format_probability_filter_cell(
                {},
                {
                    "classification_filter_label": "+3%",
                    "classification_filter_threshold": 0.6,
                    "probability_filter": {"active": True},
                },
            ),
            "3%(0.60)",
        )
        self.assertEqual(_format_probability_filter_cell({"probability_filter_active": False}), "—")
        self.assertEqual(_format_probability_filter_cell({}), "—")

    def test_active_filters_cell_probability_only(self) -> None:
        from master_dataset_tk.model_lab_strategy_sim_panel import (
            _format_active_filters_cell,
        )

        self.assertEqual(
            _format_active_filters_cell(
                {
                    "probability_filter_active": True,
                    "probability_filter_label": "+2%",
                    "probability_filter_threshold": 0.30,
                    "tb_filter_active": False,
                    "classifier_active": False,
                }
            ),
            "2%(0.30)",
        )

    def test_active_filters_cell_probability_and_tb(self) -> None:
        from master_dataset_tk.model_lab_strategy_sim_panel import (
            _format_active_filters_cell,
        )

        self.assertEqual(
            _format_active_filters_cell(
                {
                    "probability_filter_active": True,
                    "probability_filter_label": "+2%",
                    "probability_filter_threshold": 0.30,
                    "tb_filter_active": True,
                    "tb_filter_label": "TP",
                    "tb_filter_threshold": 0.60,
                }
            ),
            "2%(0.30) + TB:TP(0.60)",
        )

    def test_active_filters_cell_tb_only(self) -> None:
        from master_dataset_tk.model_lab_strategy_sim_panel import (
            _format_active_filters_cell,
        )

        self.assertEqual(
            _format_active_filters_cell(
                {
                    "probability_filter_active": False,
                    "tb_filter_active": True,
                    "tb_filter_label": "TP",
                    "tb_filter_threshold": 0.60,
                }
            ),
            "TB:TP(0.60)",
        )

    def test_active_filters_cell_includes_confidence(self) -> None:
        from master_dataset_tk.model_lab_strategy_sim_panel import (
            _format_active_filters_cell,
        )

        self.assertEqual(
            _format_active_filters_cell(
                {
                    "classifier_active": True,
                    "classifier_label": "Path Touch",
                    "classifier_filter": {"model_key": "target_hit", "keep_value": 1},
                    "probability_filter_active": True,
                    "probability_filter_label": "+2%",
                    "probability_filter_threshold": 0.30,
                    "tb_filter_active": True,
                    "tb_filter_label": "TP",
                    "tb_filter_threshold": 0.60,
                }
            ),
            "Conf:target_hit(1) + 2%(0.30) + TB:TP(0.60)",
        )

    def test_active_filters_cell_none_active(self) -> None:
        from master_dataset_tk.model_lab_strategy_sim_panel import (
            _format_active_filters_cell,
        )

        self.assertEqual(_format_active_filters_cell({}), "—")
        self.assertEqual(
            _format_active_filters_cell(
                {"probability_filter_active": False, "tb_filter_active": False, "classifier_active": False}
            ),
            "—",
        )

    def test_active_filters_cell_backward_compatible_old_run(self) -> None:
        """Old runs saved before TB/Confidence metrics existed still render correctly."""
        from master_dataset_tk.model_lab_strategy_sim_panel import (
            _format_active_filters_cell,
        )

        old_metrics = {
            "probability_filter_active": True,
            "probability_filter_label": "+3%",
            "probability_filter_threshold": 0.60,
            # No tb_filter_active / classifier_active keys at all.
        }
        self.assertEqual(_format_active_filters_cell(old_metrics), "3%(0.60)")


def _tk_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.destroy()
        return True
    except Exception:
        return False


@unittest.skipUnless(_tk_available(), "Tk display unavailable")
class PackageFilterTabTests(unittest.TestCase):
    """Prediction Thresholds tab appears only while a package member is selected."""

    def setUp(self) -> None:
        import json
        import os
        import tempfile
        import tkinter as tk

        from master_dataset_tk.model_lab_strategy_sim_panel import ModelLabStrategySimPanel

        self.tmp = tempfile.mkdtemp()
        pkg = os.path.join(self.tmp, "data", "models", "Clf_2pct")
        os.makedirs(pkg, exist_ok=True)
        with open(os.path.join(pkg, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump({
                "production_walk_forward": {
                    "threshold": 0.5,
                    "roc_auc": 0.83,
                    "threshold_analysis": [
                        {"threshold": 0.30, "precision_pct": 60.1, "recall_pct": 58.2,
                         "f1_pct": 59.2, "buy_signals": 1219},
                        {"threshold": 0.50, "precision_pct": 62.4, "recall_pct": 49.1,
                         "f1_pct": 54.9, "buy_signals": 991},
                        {"threshold": 0.70, "precision_pct": 73.8, "recall_pct": 40.0,
                         "f1_pct": 51.8, "buy_signals": 682},
                    ],
                }
            }, fh)

        self.root = tk.Tk()
        self.root.withdraw()
        self.panel = ModelLabStrategySimPanel(self.root, chart_dir=self.tmp)

    def tearDown(self) -> None:
        self.root.destroy()

    def _tab_labels(self) -> list[str]:
        return [self.panel._detail_nb.tab(t, "text") for t in self.panel._detail_nb.tabs()]

    def test_tab_hidden_until_member_selected(self) -> None:
        self.assertNotIn("Prediction Thresholds", self._tab_labels())

        self.panel._prob_options = [{
            "key": "up_2pct",
            "label": "+2% Probability",
            "ladder_label": "+2%",
            "column": "pred_prob_up_2pct_5m",
            "model_name": "Clf_2pct",
        }]
        self.panel._prob_filter_var.set("+2% Probability")
        self.panel._on_package_filter_selected()

        self.assertIn("Prediction Thresholds", self._tab_labels())
        # Sub-tab workspace: Probability Ladder / Triple Barrier / Confidence / Meta.
        subtab_labels = [
            self.panel._threshold_subtabs_nb.tab(t, "text")
            for t in self.panel._threshold_subtabs_nb.tabs()
        ]
        self.assertEqual(
            subtab_labels, ["Probability Ladder", "Triple Barrier", "Confidence", "Meta"]
        )
        # Defaults to Best Composite (0.70), not 0.50.
        self.assertEqual(self.panel._prob_threshold, 0.70)
        self.assertEqual(self.panel._prob_threshold_var.get(), "0.70")

        rows = self.panel._threshold_tree.get_children()
        self.assertEqual(list(rows), ["0.30", "0.50", "0.70"])
        selected = [
            r for r in rows
            if self.panel._threshold_tree.set(r, "selected") == "\u25cf"
        ]
        self.assertEqual(selected, ["0.70"])

        self.panel._prob_filter_var.set("Disabled")
        self.panel._on_package_filter_selected()
        self.assertNotIn("Prediction Thresholds", self._tab_labels())

    def test_clicking_row_changes_operating_threshold(self) -> None:
        self.panel._prob_options = [{
            "key": "up_2pct",
            "label": "+2% Probability",
            "ladder_label": "+2%",
            "column": "pred_prob_up_2pct_5m",
            "model_name": "Clf_2pct",
        }]
        self.panel._prob_filter_var.set("+2% Probability")
        self.panel._on_package_filter_selected()

        self.panel._threshold_tree.selection_set("0.30")
        self.panel._on_threshold_picked()
        self.assertEqual(self.panel._prob_threshold, 0.30)
        self.assertEqual(self.panel._prob_threshold_var.get(), "0.30")
        self.assertEqual(self.panel._threshold_tree.set("0.30", "selected"), "\u25cf")
        self.assertEqual(self.panel._threshold_tree.set("0.70", "selected"), "\u25cb")

    def test_tb_filter_disabled_by_default(self) -> None:
        # Part A acceptance: TB filter section exists, unchecked out of the box.
        self.assertFalse(self.panel._tb_enabled_var.get())
        kwargs = self.panel._tb_filter_kwargs_from_ui()
        self.assertFalse(kwargs["tb_filter_enabled"])
        self.assertIsNone(kwargs["tb_class_id"])
        # Class radios/threshold entry stay disabled until the checkbox is on.
        self.assertEqual(str(self.panel._tb_threshold_entry["state"]), "disabled")

    def test_tb_filter_toggle_enables_threshold_entry(self) -> None:
        self.panel._tb_classes = [
            {"class_id": 0, "label": "TP"},
            {"class_id": 1, "label": "SL"},
            {"class_id": 2, "label": "TIME"},
        ]
        self.panel._tb_enabled_var.set(True)
        self.panel._on_tb_filter_toggled()
        self.assertEqual(str(self.panel._tb_threshold_entry["state"]), "normal")
        self.panel._tb_enabled_var.set(False)
        self.panel._on_tb_filter_toggled()
        self.assertEqual(str(self.panel._tb_threshold_entry["state"]), "disabled")


@unittest.skipUnless(_tk_available(), "Tk display unavailable")
class TripleBarrierThresholdTabTests(unittest.TestCase):
    """Triple Barrier sub-tab of Prediction Thresholds — same pattern as Confidence."""

    def setUp(self) -> None:
        import json
        import os
        import tempfile
        import tkinter as tk

        from master_dataset_tk.model_lab_strategy_sim_panel import ModelLabStrategySimPanel

        self.tmp = tempfile.mkdtemp()
        pkg = os.path.join(self.tmp, "data", "models", "TBM_fixture")
        os.makedirs(pkg, exist_ok=True)
        with open(os.path.join(pkg, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "production_walk_forward": {
                        "threshold": 0.5,
                        "roc_auc": 0.81,
                        "threshold_analysis": [
                            {
                                "threshold": 0.30,
                                "precision_pct": 55.0,
                                "recall_pct": 70.0,
                                "f1_pct": 61.6,
                                "accuracy_pct": 58.0,
                                "buy_signals": 900,
                            },
                            {
                                "threshold": 0.60,
                                "precision_pct": 68.0,
                                "recall_pct": 45.0,
                                "f1_pct": 54.2,
                                "accuracy_pct": 64.0,
                                "buy_signals": 500,
                            },
                            {
                                "threshold": 0.80,
                                "precision_pct": 79.0,
                                "recall_pct": 30.0,
                                "f1_pct": 43.5,
                                "accuracy_pct": 70.0,
                                "buy_signals": 220,
                            },
                        ],
                    }
                },
                fh,
            )

        self.root = tk.Tk()
        self.root.withdraw()
        self.panel = ModelLabStrategySimPanel(self.root, chart_dir=self.tmp)

    def tearDown(self) -> None:
        self.root.destroy()

    def _load_fixture_defaults(self) -> None:
        from chain_replay_ml.strategy_simulator import resolve_member_threshold_defaults

        self.panel._tb_model_name = "TBM_fixture"
        self.panel._tb_threshold_defaults = resolve_member_threshold_defaults(
            self.panel._data_dir(), "TBM_fixture"
        )

    def test_registry_wires_a_real_builder_not_the_stub(self) -> None:
        from master_dataset_tk.model_lab_strategy_sim_panel import (
            _THRESHOLD_SUBTAB_REGISTRY,
        )

        by_title = dict(_THRESHOLD_SUBTAB_REGISTRY)
        self.assertEqual(by_title.get("Triple Barrier"), "_build_tb_threshold_subtab")

    def test_tb_threshold_tab_built_with_expected_columns(self) -> None:
        # Built unconditionally during _build_ui — independent of package filter
        # selection (unlike the Confidence sub-tab table).
        self.assertTrue(hasattr(self.panel, "_tb_threshold_tree"))
        self.assertEqual(
            list(self.panel._tb_threshold_tree["columns"]),
            ["thr", "precision", "recall", "f1", "accuracy", "buy", "tpd", "composite", "selected"],
        )

    def test_no_tb_model_shows_clear_empty_state(self) -> None:
        self.panel._tb_model_name = None
        self.panel._tb_threshold_defaults = {}
        self.panel._render_tb_threshold_table()
        self.assertEqual(len(self.panel._tb_threshold_tree.get_children()), 0)
        hint = self.panel._tb_threshold_hint_var.get()
        self.assertIn("No Triple Barrier model linked", hint)
        self.assertNotIn("coming soon", hint.lower())

    def test_missing_metrics_json_shows_clear_empty_state(self) -> None:
        from chain_replay_ml.strategy_simulator import resolve_member_threshold_defaults

        self.panel._tb_model_name = "NoSuchTbModel"
        self.panel._tb_threshold_defaults = resolve_member_threshold_defaults(
            self.panel._data_dir(), "NoSuchTbModel"
        )
        self.panel._render_tb_threshold_table()
        self.assertEqual(len(self.panel._tb_threshold_tree.get_children()), 0)
        hint = self.panel._tb_threshold_hint_var.get()
        self.assertIn("NoSuchTbModel", hint)
        self.assertIn("No threshold analysis found", hint)
        self.assertNotIn("coming soon", hint.lower())

    def test_loads_rows_from_metrics_fixture_not_recomputed(self) -> None:
        self._load_fixture_defaults()
        self.panel._render_tb_threshold_table()

        rows = self.panel._tb_threshold_tree.get_children()
        self.assertEqual(list(rows), ["0.30", "0.60", "0.80"])

        # Row values come straight from the persisted metrics.json — never
        # recomputed / re-inferred.
        vals = self.panel._tb_threshold_tree.item("0.80")["values"]
        thr_txt, precision, recall, f1, accuracy, buy, tpd, composite, selected = vals
        self.assertEqual(thr_txt, "0.80")
        self.assertEqual(precision, "79.00%")
        self.assertEqual(accuracy, "70.00%")
        self.assertEqual(str(buy), "220")

        # Recommended (Best Composite) row is starred.
        recommended = self.panel._tb_threshold_defaults.get("recommended_threshold")
        self.assertEqual(recommended, 0.60)
        rec_vals = self.panel._tb_threshold_tree.item(f"{recommended:.2f}")["values"]
        self.assertEqual(rec_vals[0], "0.60 \u2605")

    def test_selecting_threshold_persists_for_tb_filter(self) -> None:
        from master_dataset_tk.strategy_sim_prefs import load_strategy_sim_prefs

        self._load_fixture_defaults()
        self.panel._render_tb_threshold_table()

        self.panel._tb_threshold_tree.selection_set("0.30")
        self.panel._on_tb_threshold_row_picked()

        # Same field the TB Filter's "Minimum Probability" entry reads from.
        self.assertEqual(self.panel._tb_threshold_var.get(), "0.30")
        self.assertEqual(self.panel._tb_filter_kwargs_from_ui()["tb_threshold"], 0.30)
        self.assertEqual(
            self.panel._tb_threshold_tree.set("0.30", "selected"), "\u25cf"
        )

        prefs = load_strategy_sim_prefs(self.tmp)
        tb_prefs = prefs.get("triple_barrier_filter") or {}
        self.assertEqual(tb_prefs.get("threshold"), 0.30)

    def test_prediction_thresholds_tab_shown_when_only_tb_model_linked(self) -> None:
        """Unlike Confidence, TB linkage alone (no package member) should show the tab."""
        self.panel._tb_model_name = "TBM_fixture"
        self.panel._sync_threshold_tab()
        labels = [
            self.panel._detail_nb.tab(t, "text") for t in self.panel._detail_nb.tabs()
        ]
        self.assertIn("Prediction Thresholds", labels)


if __name__ == "__main__":
    unittest.main()
