"""Shared types for the Outcome Label Engine (Phase 1 foundation)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ProblemType = Literal[
    "regression",
    "binary_classification",
    "multiclass",
    "ranking",
    "probability",
]

SOURCE_MASTER = "master"
SOURCE_PREDICTION = "prediction"

ENGINE_VERSION = "6.1"

# Config-schema field types understood by UI renderers.
ConfigFieldType = Literal[
    "int",
    "float",
    "bool",
    "str",
    "enum",
    "int_list",
    "float_list",
]


@dataclass(frozen=True)
class StrategyMetadata:
    """UI-facing copy — selectors render this; never hardcode titles in Model Builder."""

    strategy_id: str
    version: str
    display_name: str
    description: str
    category: str


@dataclass(frozen=True)
class StrategyCapabilities:
    strategy_id: str
    supported_sources: frozenset[str]
    supported_problem_types: frozenset[ProblemType]


@dataclass(frozen=True)
class LabelStrategyConfig:
    strategy_id: str
    version: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelSourceContext:
    """Which read-only source feeds this labeling run (handles are opaque)."""

    source_kind: str
    day: str | None = None
    handles: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetDefinitions:
    """Train/display columns + encoding — training never guesses."""

    primary_target: str
    display_target: str | None = None
    label_encoding: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_target": self.primary_target,
            "display_target": self.display_target,
            "label_encoding": (
                dict(self.label_encoding) if self.label_encoding is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetDefinitions:
        enc = data.get("label_encoding")
        return cls(
            primary_target=str(data["primary_target"]),
            display_target=data.get("display_target"),
            label_encoding=dict(enc) if enc is not None else None,
        )


@dataclass
class LabelBatchResult:
    rows: list[dict[str, Any]]
    target_columns: list[str]
    target_definitions: TargetDefinitions
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelRunMeta:
    """Persisted beside each immutable Training Dataset artifact."""

    strategy: str
    version: str
    engine_version: str
    source: str
    params: dict[str, Any]
    rows: int
    compute_time_sec: float
    supported_problem_types: list[str]
    target_columns: list[str]
    target_definitions: dict[str, Any]
    days_processed: list[str] = field(default_factory=list)
    valid_rows: int | None = None
    invalid_rows: int | None = None
    created_at_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabelRunMeta:
        return cls(
            strategy=str(data["strategy"]),
            version=str(data["version"]),
            engine_version=str(data["engine_version"]),
            source=str(data["source"]),
            params=dict(data.get("params") or {}),
            rows=int(data["rows"]),
            compute_time_sec=float(data["compute_time_sec"]),
            supported_problem_types=list(data.get("supported_problem_types") or []),
            target_columns=list(data.get("target_columns") or []),
            target_definitions=dict(data.get("target_definitions") or {}),
            days_processed=list(data.get("days_processed") or []),
            valid_rows=data.get("valid_rows"),
            invalid_rows=data.get("invalid_rows"),
            created_at_utc=data.get("created_at_utc"),
        )


def defaults_from_config_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Extract default param values from a strategy config schema."""
    out: dict[str, Any] = {}
    for name, spec in schema.items():
        if not isinstance(spec, dict):
            raise TypeError(f"config schema field {name!r} must be a dict")
        if "default" in spec:
            out[name] = spec["default"]
    return out


def normalize_enum_choices(choices: Any) -> list[dict[str, Any]]:
    """Normalize schema ``choices`` to ``[{value, label, enabled}, …]``."""
    out: list[dict[str, Any]] = []
    for raw in choices or []:
        if isinstance(raw, str):
            out.append({"value": raw, "label": raw, "enabled": True})
            continue
        if not isinstance(raw, dict):
            continue
        value = str(raw.get("value") if "value" in raw else raw.get("id") or "").strip()
        if not value:
            continue
        out.append(
            {
                "value": value,
                "label": str(raw.get("label") or value),
                "enabled": bool(raw.get("enabled", True)),
            }
        )
    return out


def validate_config_against_schema(
    params: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Merge defaults and ensure params only use known schema keys.

    Validates ``enum`` fields against ``choices`` (rejects disabled values).
    """
    merged = defaults_from_config_schema(schema)
    unknown = set(params) - set(schema)
    if unknown:
        raise ValueError(f"unknown config params: {sorted(unknown)}")
    merged.update(params)
    for name, spec in schema.items():
        if not isinstance(spec, dict):
            continue
        if str(spec.get("type") or "") != "enum":
            continue
        choices = normalize_enum_choices(spec.get("choices"))
        by_value = {c["value"]: c for c in choices}
        val = merged.get(name)
        key = str(val) if val is not None else ""
        if key not in by_value:
            allowed = sorted(by_value)
            raise ValueError(f"invalid {name}={val!r}; expected one of {allowed}")
        if not by_value[key].get("enabled", True):
            raise ValueError(f"{name}={val!r} is not enabled yet")
        merged[name] = key
    return merged
