"""Tests for feature project organization fields."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.dataset_builder.feature_domains import DOMAIN_LABELS, DOMAIN_ORDER
from chain_replay_ml.dataset_builder.feature_project_organization import (
    backfill_feature_group_map,
    canonical_group_for_feature,
    canonical_registry_groups,
    is_canonical_domain_id,
    migrate_project_organization,
    normalize_custom_project_groups,
    project_group_tree,
)
from chain_replay_ml.dataset_builder.feature_registry_store import (
    create_project,
    load_store,
    suggest_project_id,
)
from master_dataset_tk.feature_selection_engine import (
    backfill_feature_group_map as engine_backfill,
    project_group_options,
    registry_group_for_feature,
)


class FeatureProjectOrganizationTests(unittest.TestCase):
    def test_suggest_project_id_unique(self) -> None:
        store = load_store(self._tmpdir)
        first = suggest_project_id(store, "Chart")
        store_projects = dict(store.get("projects") or {})
        store_projects[first] = {"label": "Chart"}
        store["projects"] = store_projects
        second = suggest_project_id(store, "Chart")
        self.assertNotEqual(first, second)

    def test_create_project_stores_group_map(self) -> None:
        doc = create_project(
            self._tmpdir,
            label="Test",
            project_id="test_proj",
            feature_names=["ltp", "delta"],
            project_groups=[{"id": "momentum", "label": "Momentum"}],
            feature_group_map={"ltp": "momentum"},
        )
        self.assertEqual(doc["id"], "test_proj")
        self.assertEqual((doc.get("feature_group_map") or {}).get("ltp"), "momentum")
        self.assertEqual((doc.get("feature_group_map") or {}).get("delta"), "greeks")

    def test_canonical_group_for_delta(self) -> None:
        self.assertEqual(canonical_group_for_feature("delta"), "greeks")
        self.assertEqual(registry_group_for_feature({}, "delta"), "greeks")

    def test_backfill_uses_canonical_domain(self) -> None:
        out = backfill_feature_group_map(
            ["delta"],
            project_groups=[],
            feature_group_map={},
        )
        self.assertEqual(out.get("delta"), "greeks")
        engine_out = engine_backfill(
            {},
            ["delta"],
            project_groups=[],
            feature_group_map={},
        )
        self.assertEqual(engine_out.get("delta"), "greeks")

    def test_normalize_strips_canonical_from_project_groups(self) -> None:
        groups = normalize_custom_project_groups(
            [
                {"id": "greeks", "label": "Greeks"},
                {"id": "momentum", "label": "Momentum"},
            ]
        )
        ids = {g["id"] for g in groups}
        self.assertNotIn("greeks", ids)
        self.assertIn("momentum", ids)

    def test_project_group_tree_matches_canonical_labels(self) -> None:
        rows = project_group_tree(project_groups=[], feature_group_map={})
        labels = [r["label"] for r in rows]
        for domain_id in DOMAIN_ORDER:
            self.assertIn(DOMAIN_LABELS[domain_id], labels)

    def test_project_group_options_matches_tree(self) -> None:
        opts = project_group_options({}, [], {})
        labels = [o["label"] for o in opts]
        self.assertIn("Price & Premium", labels)
        self.assertIn("Greeks", labels)

    def test_migrate_legacy_schema_group_to_domain(self) -> None:
        migrated = migrate_project_organization(
            {
                "feature_names": ["delta"],
                "project_groups": [],
                "feature_group_map": {"delta": "legacy_schema_greeks"},
            }
        )
        self.assertEqual(migrated["feature_group_map"].get("delta"), "greeks")
        self.assertEqual(migrated["group_ids"], ["greeks"])

    def test_migrate_preserves_user_custom_group(self) -> None:
        migrated = migrate_project_organization(
            {
                "feature_names": ["delta"],
                "project_groups": [{"id": "momentum", "label": "Momentum"}],
                "feature_group_map": {"delta": "momentum"},
            }
        )
        self.assertEqual(migrated["feature_group_map"].get("delta"), "momentum")
        ids = {g["id"] for g in migrated.get("project_groups") or []}
        self.assertIn("momentum", ids)

    def test_is_canonical_domain_id(self) -> None:
        self.assertTrue(is_canonical_domain_id("greeks"))
        self.assertFalse(is_canonical_domain_id("momentum"))

    def test_canonical_registry_groups_have_features(self) -> None:
        groups = canonical_registry_groups()
        self.assertTrue(any(g.get("label") == "Greeks" for g in groups))
        greeks = next(g for g in groups if g.get("id") == "greeks")
        self.assertIn("delta", greeks.get("features") or [])

    def test_ensure_all_project_creates_with_active_features(self) -> None:
        from chain_replay_ml.dataset_builder.feature_project_organization import (
            RESERVED_ALL_PROJECT_ID,
            canonical_group_for_feature,
        )
        from chain_replay_ml.dataset_builder.feature_registry_store import ensure_all_project

        doc = ensure_all_project(self._tmpdir)
        self.assertEqual(doc["id"], RESERVED_ALL_PROJECT_ID)
        self.assertIn("ltp", doc.get("feature_names") or [])
        self.assertEqual(
            (doc.get("feature_group_map") or {}).get("delta"),
            canonical_group_for_feature("delta"),
        )

    def test_delete_all_project_blocked(self) -> None:
        from chain_replay_ml.dataset_builder.feature_registry_store import (
            delete_project,
            ensure_all_project,
        )

        ensure_all_project(self._tmpdir)
        with self.assertRaises(ValueError):
            delete_project(self._tmpdir, "all")

    def test_create_project_blocks_all_id(self) -> None:
        from chain_replay_ml.dataset_builder.feature_registry_store import create_project

        with self.assertRaises(ValueError):
            create_project(self._tmpdir, label="All", project_id="all")

    def test_project_registry_feature_source_uses_custom_groups(self) -> None:
        from chain_replay_ml.dataset_builder.feature_project_organization import (
            project_registry_feature_source,
            RESERVED_ALL_PROJECT_ID,
        )
        from chain_replay_ml.dataset_builder.feature_registry_store import (
            create_project,
            ensure_all_project,
            update_project,
        )

        ensure_all_project(self._tmpdir)
        update_project(
            self._tmpdir,
            RESERVED_ALL_PROJECT_ID,
            project_groups=[{"id": "price", "label": "Price"}],
            feature_group_map={"ltp": "price"},
            feature_names=["ltp", "delta"],
        )
        source = project_registry_feature_source(
            data_dir=self._tmpdir,
            project_id=RESERVED_ALL_PROJECT_ID,
        )
        labels = [g.get("label") for g in source.get("groups") or []]
        self.assertIn("Price", labels)
        price = next(g for g in source["groups"] if g.get("label") == "Price")
        self.assertIn("ltp", price.get("features") or [])
        self.assertNotIn("ltp", next(g for g in source["groups"] if g.get("id") == "greeks").get("features") or [])

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        path = os.path.join(self._tmpdir, "feature_registry_store.json")
        if os.path.isfile(path):
            os.remove(path)
        os.rmdir(self._tmpdir)


if __name__ == "__main__":
    unittest.main()
