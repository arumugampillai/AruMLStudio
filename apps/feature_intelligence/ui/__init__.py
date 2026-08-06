"""ui package — Feature Intelligence Studio panels (Sprint 9)."""

from __future__ import annotations

from feature_intelligence.ui.feature_inspector_panel import (
    FeatureInspectorPanel,
    FeatureIntelligenceSearchBar,
)
from feature_intelligence.ui.inspect_format import (
    SearchPlan,
    build_search_plan,
    filter_hits_by_plan,
    header_summary_lines,
    lineage_tree_text,
)

__all__ = [
    "FeatureInspectorPanel",
    "FeatureIntelligenceSearchBar",
    "SearchPlan",
    "build_search_plan",
    "filter_hits_by_plan",
    "header_summary_lines",
    "lineage_tree_text",
]
