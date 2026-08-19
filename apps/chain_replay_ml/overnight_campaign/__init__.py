"""Phase 4F.5: Autonomous Overnight Research Campaign Controller."""

from .persistence import (
    init_campaign_tables,
    load_campaign_state,
    persist_campaign_state,
)
from .runner import OvernightCampaignRunner
from .types import (
    CampaignConfig,
    CampaignState,
    CampaignStatus,
    CampaignStopReason,
    OvernightCampaignReport,
)

__all__ = [
    "CampaignConfig",
    "CampaignState",
    "CampaignStatus",
    "CampaignStopReason",
    "OvernightCampaignReport",
    "OvernightCampaignRunner",
    "init_campaign_tables",
    "load_campaign_state",
    "persist_campaign_state",
]
