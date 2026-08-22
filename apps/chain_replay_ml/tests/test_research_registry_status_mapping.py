"""Regression test for Research Registry status mapping & historical synchronization."""

import unittest

from chain_replay_ml.core.data_root import DataRootService
from chain_replay_ml.research_registry.store import (
    backfill_historical_research_records,
    get_all_research_records,
    get_research_detail,
    map_campaign_status_to_research_status,
)
from chain_replay_ml.research_registry.types import ResearchStatus


class TestResearchRegistryStatusMapping(unittest.TestCase):
    """Verify case-insensitive status normalization and registry lifecycle state fidelity."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = DataRootService().get_data_root()

    def test_status_mapping_normalization(self) -> None:
        """Verify map_campaign_status_to_research_status handles uppercase, lowercase, and edge cases."""
        # 1. Normal completions
        self.assertEqual(map_campaign_status_to_research_status("COMPLETED"), ResearchStatus.COMPLETED)
        self.assertEqual(map_campaign_status_to_research_status("completed"), ResearchStatus.COMPLETED)
        self.assertEqual(map_campaign_status_to_research_status("Completed"), ResearchStatus.COMPLETED)
        self.assertEqual(map_campaign_status_to_research_status("CAMPAIGN_COMPLETED"), ResearchStatus.COMPLETED)

        # 2. Failures
        self.assertEqual(map_campaign_status_to_research_status("CAMPAIGN_FAILED"), ResearchStatus.FAILED)
        self.assertEqual(map_campaign_status_to_research_status("campaign_failed"), ResearchStatus.FAILED)
        self.assertEqual(map_campaign_status_to_research_status("FAILED"), ResearchStatus.FAILED)

        # 3. Active execution
        self.assertEqual(map_campaign_status_to_research_status("RUNNING"), ResearchStatus.RUNNING)
        self.assertEqual(map_campaign_status_to_research_status("running"), ResearchStatus.RUNNING)

        # 4. Pauses
        self.assertEqual(map_campaign_status_to_research_status("RESOURCE_PAUSED"), ResearchStatus.PAUSED)
        self.assertEqual(map_campaign_status_to_research_status("resource_paused"), ResearchStatus.PAUSED)

        # 5. Interrupted or unexecuted stubs -> ABORTED
        self.assertEqual(map_campaign_status_to_research_status("CREATED"), ResearchStatus.ABORTED)
        self.assertEqual(map_campaign_status_to_research_status("created"), ResearchStatus.ABORTED)
        self.assertEqual(map_campaign_status_to_research_status("TRAINING"), ResearchStatus.ABORTED)
        self.assertEqual(map_campaign_status_to_research_status("CAMPAIGN_STOPPED"), ResearchStatus.ABORTED)

        # 6. User cancellation stop reason override
        self.assertEqual(map_campaign_status_to_research_status("COMPLETED", "USER_CANCELLED"), ResearchStatus.ABORTED)
        self.assertEqual(map_campaign_status_to_research_status("RUNNING", "user_cancelled"), ResearchStatus.ABORTED)

    def test_historical_status_distribution(self) -> None:
        """Verify historical research runs in analysis.db match authoritative distribution."""
        backfill_historical_research_records(self.data_dir, force_resync=False)
        records = get_all_research_records(self.data_dir)
        self.assertEqual(len(records), 34)

        status_counts = {}
        for r in records:
            st = r.get("status")
            status_counts[st] = status_counts.get(st, 0) + 1

        self.assertEqual(status_counts.get(ResearchStatus.COMPLETED.value), 17)
        self.assertEqual(status_counts.get(ResearchStatus.FAILED.value), 9)
        self.assertEqual(status_counts.get(ResearchStatus.ABORTED.value), 8)
        self.assertEqual(status_counts.get(ResearchStatus.RUNNING.value, 0), 0)

    def test_specific_completed_research_run(self) -> None:
        """Verify the multi-generation research run with 56 features is marked COMPLETED."""
        r_id = "RESEARCH_NIFTY_6_standard_all_20260821_185913_37f9"
        detail = get_research_detail(self.data_dir, r_id)
        self.assertIsNotNone(detail)
        self.assertEqual(detail["status"], ResearchStatus.COMPLETED.value)
        self.assertEqual(detail["stop_reason"], "MAX_GENERATIONS_REACHED")
        self.assertEqual(detail["total_df_features_created"], 56)
        self.assertEqual(detail["keep_count"], 9)
        self.assertEqual(detail["watch_count"], 5)
        self.assertEqual(detail["remove_count"], 42)
        self.assertEqual(detail["best_candidate_id"], "CAND_NIFTY_XGB_42158c39_G1_XGB_SHAP")


if __name__ == "__main__":
    unittest.main()
