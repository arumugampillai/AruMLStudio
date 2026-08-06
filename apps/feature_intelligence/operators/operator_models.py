"""Operator Registry models (Sprint 3)."""

from __future__ import annotations

from dataclasses import dataclass

PARAMETER_TYPES = frozenset(
    {"feature", "primitive", "integer", "float", "boolean", "string", "window", "list"}
)
CATEGORIES = frozenset(
    {
        "TREND",
        "ROLLING",
        "STATISTICAL",
        "ARITHMETIC",
        "NORMALIZATION",
        "TIME",
        "COMPARISON",
        "TRANSFORMATION",
        "INTERACTION",
        "OTHER",
    }
)
WARMUP_POLICIES = frozenset({"NONE", "WINDOW", "CUSTOM"})
MISSING_DATA_POLICIES = frozenset(
    {"DROP", "FORWARD_FILL", "BACKWARD_FILL", "ZERO", "NAN", "ERROR"}
)
COMPLEXITY_CLASSES = frozenset({"O(1)", "O(n)", "O(window)", "O(log n)"})


@dataclass(frozen=True)
class OperatorRecord:
    operator_id: str
    canonical_name: str
    display_name: str
    category: str
    formula: str
    definition_text: str
    parameter_schema_json: str
    input_arity_min: int
    input_arity_max: int | None
    output_count: int
    warmup_policy: str
    missing_data_policy: str
    deterministic: bool
    stateful: bool
    streaming_supported: bool
    incremental_supported: bool
    complexity_class: str
    operator_version: str
    catalog_version: str
    operator_pack_version: str
    description: str | None = None
    depends_on_operator_ids: str | None = None
    extras_json: str | None = None
    created_at: str = ""
    updated_at: str = ""
