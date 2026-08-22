"""Autonomous Research Registry Package (Doc 16)."""

from .memory import (
    get_blacklisted_formula_hashes,
    update_formula_memory_from_discovery,
)
from .store import (
    backfill_historical_research_records,
    generate_research_id,
    get_all_research_records,
    get_research_detail,
    init_research_registry_tables,
    insert_or_update_research_run,
    record_generation_linkage,
)
from .types import (
    FormulaGlobalStatus,
    FormulaMemoryRecord,
    ResearchGenerationLinkage,
    ResearchRegistryRecord,
    ResearchStatus,
)

__all__ = [
    "ResearchStatus",
    "FormulaGlobalStatus",
    "ResearchRegistryRecord",
    "ResearchGenerationLinkage",
    "FormulaMemoryRecord",
    "generate_research_id",
    "init_research_registry_tables",
    "insert_or_update_research_run",
    "record_generation_linkage",
    "get_all_research_records",
    "get_research_detail",
    "backfill_historical_research_records",
    "get_blacklisted_formula_hashes",
    "update_formula_memory_from_discovery",
]
