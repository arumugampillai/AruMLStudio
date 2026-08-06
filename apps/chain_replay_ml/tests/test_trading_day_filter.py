"""Tests for trading-day filter helpers."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.trading_day_filter import (
    MODE_ALL,
    MODE_EXCLUDE_EXPIRY,
    MODE_EXPIRY_ONLY,
    apply_trading_day_filter,
    day_has_tag,
    normalize_mode,
    register_day_tag,
)


class TestTradingDayFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.days = {
            "2026-01-02": {"trading_day": "2026-01-02", "is_expiry_day": 0},
            "2026-01-08": {"trading_day": "2026-01-08", "is_expiry_day": 1},
            "2026-01-09": {"trading_day": "2026-01-09", "is_expiry_day": 0},
            "2026-01-15": {"trading_day": "2026-01-15", "is_expiry_day": 1},
        }
        self.selected = list(self.days)

    def test_all_preserves_selection(self) -> None:
        exported, meta = apply_trading_day_filter(self.selected, self.days, MODE_ALL)
        self.assertEqual(exported, sorted(self.selected))
        self.assertEqual(meta["mode"], "all")
        self.assertEqual(meta["selected_days"], 4)
        self.assertEqual(meta["exported_days"], 4)
        self.assertEqual(meta["excluded_dates"], [])
        self.assertEqual(meta["expiry_dates"], ["2026-01-08", "2026-01-15"])
        self.assertEqual(meta["exported_dates"], sorted(self.selected))

    def test_exclude_expiry(self) -> None:
        exported, meta = apply_trading_day_filter(self.selected, self.days, MODE_EXCLUDE_EXPIRY)
        self.assertEqual(exported, ["2026-01-02", "2026-01-09"])
        self.assertEqual(meta["mode"], "exclude_expiry")
        self.assertEqual(meta["selected_days"], 4)
        self.assertEqual(meta["exported_days"], 2)
        self.assertEqual(meta["excluded_dates"], ["2026-01-08", "2026-01-15"])
        self.assertEqual(meta["expiry_dates"], ["2026-01-08", "2026-01-15"])

    def test_expiry_only(self) -> None:
        exported, meta = apply_trading_day_filter(self.selected, self.days, MODE_EXPIRY_ONLY)
        self.assertEqual(exported, ["2026-01-08", "2026-01-15"])
        self.assertEqual(meta["exported_days"], 2)
        self.assertEqual(meta["excluded_dates"], ["2026-01-02", "2026-01-09"])
        self.assertEqual(meta["expiry_dates"], ["2026-01-08", "2026-01-15"])

    def test_dominant_expiry_fallback(self) -> None:
        day = {"trading_day": "2026-01-08", "dominant_expiry": "2026-01-08"}
        self.assertTrue(day_has_tag(day, "expiry"))

    def test_register_custom_tag(self) -> None:
        register_day_tag("holiday", lambda d: bool(d.get("is_holiday")))
        self.assertTrue(day_has_tag({"is_holiday": True}, "holiday"))
        self.assertFalse(day_has_tag({}, "holiday"))

    def test_exclude_expiry_backfill_from_master_days(self) -> None:
        from chain_replay_ml.dataset_builder.trading_day_filter import (
            enrich_trading_day_filter_dates,
            trading_day_filter_summary_rows,
        )

        enriched = enrich_trading_day_filter_dates(
            {
                "mode": "exclude_expiry",
                "selected_days": 4,
                "exported_days": 2,
            },
            exported_dates=["2026-01-02", "2026-01-09"],
            master_day_rows=[
                {"trading_day": "2026-01-02", "is_expiry_day": 0},
                {"trading_day": "2026-01-08", "is_expiry_day": 1},
                {"trading_day": "2026-01-09", "is_expiry_day": 0},
                {"trading_day": "2026-01-15", "is_expiry_day": 1},
            ],
        )
        self.assertEqual(enriched["excluded_dates"], ["2026-01-08", "2026-01-15"])
        rows = trading_day_filter_summary_rows(
            {
                "mode": "exclude_expiry",
                "selected_days": 4,
                "exported_days": 2,
            },
            exported_dates=["2026-01-02", "2026-01-09"],
            master_day_rows=[
                {"trading_day": "2026-01-02", "is_expiry_day": 0},
                {"trading_day": "2026-01-08", "is_expiry_day": 1},
                {"trading_day": "2026-01-09", "is_expiry_day": 0},
                {"trading_day": "2026-01-15", "is_expiry_day": 1},
            ],
        )
        by_label = {r["label"]: r["value"] for r in rows}
        self.assertEqual(by_label["Excluded expiry dates"], "2026-01-08, 2026-01-15")


class TestResolveDayScopeFilter(unittest.TestCase):
    """Feature Transformation → Auto: All days / Selected days scope resolution."""

    def setUp(self) -> None:
        self.master_days = ["2026-06-30", "2026-07-07", "2026-07-14", "2026-07-21"]

    def test_all_days_resolves_concrete_master_dates(self) -> None:
        from chain_replay_ml.dataset_builder.trading_day_filter import resolve_day_scope_filter

        all_days, explicit_days, meta = resolve_day_scope_filter(
            scope="all",
            selected_days=set(),
            master_days=self.master_days,
        )
        self.assertTrue(all_days)
        self.assertEqual(explicit_days, [])
        self.assertEqual(meta["mode"], "all")
        self.assertEqual(meta["selected_dates"], sorted(self.master_days))
        self.assertEqual(meta["exported_dates"], sorted(self.master_days))
        self.assertEqual(meta["selected_days"], len(self.master_days))
        self.assertEqual(meta["exported_days"], len(self.master_days))

    def test_selected_days_resolves_to_explicit_subset(self) -> None:
        from chain_replay_ml.dataset_builder.trading_day_filter import resolve_day_scope_filter

        chosen = {"2026-07-21", "2026-06-30"}
        all_days, explicit_days, meta = resolve_day_scope_filter(
            scope="selected",
            selected_days=chosen,
            master_days=self.master_days,
        )
        self.assertFalse(all_days)
        self.assertEqual(explicit_days, ["2026-06-30", "2026-07-21"])
        self.assertEqual(meta["selected_dates"], ["2026-06-30", "2026-07-21"])
        self.assertEqual(meta["exported_dates"], ["2026-06-30", "2026-07-21"])
        self.assertEqual(meta["selected_days"], 2)
        self.assertEqual(meta["exported_days"], 2)

    def test_selected_days_drops_dates_not_in_master(self) -> None:
        from chain_replay_ml.dataset_builder.trading_day_filter import resolve_day_scope_filter

        all_days, explicit_days, meta = resolve_day_scope_filter(
            scope="selected",
            selected_days={"2026-07-07", "2099-01-01"},
            master_days=self.master_days,
        )
        self.assertFalse(all_days)
        self.assertEqual(explicit_days, ["2026-07-07"])
        self.assertEqual(meta["selected_dates"], ["2026-07-07"])

    def test_selected_days_empty_selection_yields_no_dates(self) -> None:
        from chain_replay_ml.dataset_builder.trading_day_filter import resolve_day_scope_filter

        all_days, explicit_days, meta = resolve_day_scope_filter(
            scope="selected",
            selected_days=set(),
            master_days=self.master_days,
        )
        self.assertFalse(all_days)
        self.assertEqual(explicit_days, [])
        self.assertEqual(meta["selected_dates"], [])

    def test_master_trading_days_missing_file_returns_empty(self) -> None:
        from chain_replay_ml.dataset_builder.trading_day_filter import master_trading_days

        self.assertEqual(master_trading_days(""), [])
        self.assertEqual(master_trading_days("/no/such/master.db"), [])

    def test_master_trading_days_reads_from_store(self) -> None:
        import os
        import tempfile

        from chain_replay_ml.dataset_builder.master_store import MasterStore
        from chain_replay_ml.dataset_builder.trading_day_filter import master_trading_days

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "master.db")
            store = MasterStore(db_path)
            store.open()
            try:
                store.conn.executemany(
                    "INSERT INTO master_dataset_days (trading_day, row_count) VALUES (?, ?)",
                    [("2026-07-21", 10), ("2026-06-30", 8)],
                )
                store.conn.commit()
            finally:
                store.close()
            self.assertEqual(master_trading_days(db_path), ["2026-06-30", "2026-07-21"])


if __name__ == "__main__":
    unittest.main()
