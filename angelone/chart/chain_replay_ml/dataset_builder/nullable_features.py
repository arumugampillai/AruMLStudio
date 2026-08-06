"""Central Nullable Feature List for Dataset Builder No-Null Step 2.

By default every feature is Non-Nullable. Features listed in
``NULLABLE_FEATURE_LIST`` are **Explicit Nullable** (Registry policy).

Pipeline / interaction outputs that depend on any nullable parent become
**Inherited Nullable** via the transformation dependency graph. Step 2 ignores
both Explicit and Inherited nullable columns when dropping incomplete rows.

Step 1 (drop 100% NULL columns) is unchanged and does not consult this list.

Add or remove **explicit** names here only. Do not hand-maintain downstream
pipeline names — inheritance covers them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# Single source of truth for Explicit Nullable (Registry). Order is display-only.
NULLABLE_FEATURE_LIST: tuple[str, ...] = (
    "gamma_flip_spot",
    # Companion of gamma_flip_spot: always NULL when flip is unresolved.
    "gamma_flip_distance",
    # Option IV / higher-order greeks: can be unresolved for some contracts/ticks.
    "current_iv",
    "vega",
    "vanna",
    "charm",
    "speed",
)

_NULLABLE_SET: frozenset[str] = frozenset(NULLABLE_FEATURE_LIST)


@dataclass(frozen=True)
class NullableResolution:
    """Explicit Registry nullable + Inherited pipeline nullable.

    ``inherits_from`` maps each Inherited Nullable child to the immediate
    nullable parent name(s) that caused the classification (traceable reason).
    """

    explicit: frozenset[str]
    inherited: frozenset[str]
    inherits_from: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def effective(self) -> frozenset[str]:
        return self.explicit | self.inherited

    def present_explicit(self, column_names: Sequence[str]) -> list[str]:
        names = {str(c).strip() for c in column_names if str(c).strip()}
        return [c for c in NULLABLE_FEATURE_LIST if c in names and c in self.explicit]

    def present_inherited(self, column_names: Sequence[str]) -> list[str]:
        names = {str(c).strip() for c in column_names if str(c).strip()}
        return sorted(c for c in self.inherited if c in names)

    def present_all(self, column_names: Sequence[str]) -> list[str]:
        return self.present_explicit(column_names) + self.present_inherited(column_names)

    def inheritance_parents(self, name: str) -> tuple[str, ...]:
        """Immediate nullable parents that caused Inherited status for ``name``."""
        return tuple(self.inherits_from.get(str(name or "").strip()) or ())


def nullable_feature_names() -> frozenset[str]:
    """Return the Explicit Nullable Feature List as a frozenset."""
    return _NULLABLE_SET


def is_nullable_feature(name: str) -> bool:
    """True if ``name`` is Explicit Nullable (Registry list only)."""
    return str(name or "").strip() in _NULLABLE_SET


def is_effectively_nullable(
    name: str,
    *,
    resolution: NullableResolution | None = None,
) -> bool:
    n = str(name or "").strip()
    if not n:
        return False
    if resolution is None:
        return n in _NULLABLE_SET
    return n in resolution.effective


def expand_nullable_via_lineage(
    lineage: Mapping[str, Mapping[str, Any]] | None,
    *,
    explicit: frozenset[str] | None = None,
    column_names: Sequence[str] | None = None,
) -> NullableResolution:
    """Propagate Explicit Nullable through parent→child dependency edges.

    Rule: if any parent is nullable (explicit or already inherited), the child
    is Inherited Nullable. Runs to fixpoint so chains propagate.

    Each inherited child records the immediate nullable parent(s) in
    ``inherits_from`` (not the full root chain).
    """
    base = _NULLABLE_SET if explicit is None else frozenset(explicit)
    if not lineage:
        scope = (
            {str(c).strip() for c in (column_names or []) if str(c).strip()}
            if column_names is not None
            else None
        )
        explicit_scoped = base if scope is None else frozenset(c for c in base if c in scope)
        return NullableResolution(
            explicit=explicit_scoped,
            inherited=frozenset(),
            inherits_from={},
        )

    # Restrict propagation to known columns when provided (export frame / step2 set).
    scope: set[str] | None = None
    if column_names is not None:
        scope = {str(c).strip() for c in column_names if str(c).strip()}
        # Parents outside scope still count if they are explicit nullable
        # (e.g. current_iv on frame while classifying pipeline cols only).
        scope |= {str(c).strip() for c in base if str(c).strip()}

    effective: set[str] = set(base)
    inherited: set[str] = set()
    inherits_from: dict[str, tuple[str, ...]] = {}
    # Fixpoint — order of lineage entries is arbitrary.
    changed = True
    while changed:
        changed = False
        for raw_name, info in lineage.items():
            child = str(raw_name or "").strip()
            if not child or child in effective:
                continue
            if scope is not None and child not in scope:
                continue
            if not isinstance(info, Mapping):
                continue
            parents = [
                str(p).strip()
                for p in (info.get("parents") or [])
                if str(p).strip()
            ]
            if not parents:
                continue
            nullable_parents = tuple(p for p in parents if p in effective)
            if nullable_parents:
                effective.add(child)
                if child not in base:
                    inherited.add(child)
                    inherits_from[child] = nullable_parents
                changed = True

    explicit_out = frozenset(c for c in base if scope is None or c in scope)
    return NullableResolution(
        explicit=explicit_out,
        inherited=frozenset(inherited),
        inherits_from=dict(inherits_from),
    )


def resolve_nullable_features(
    column_names: Sequence[str],
    *,
    transformation_config: dict[str, Any] | None = None,
    lineage: Mapping[str, Mapping[str, Any]] | None = None,
    explicit: frozenset[str] | None = None,
) -> NullableResolution:
    """Resolve Explicit + Inherited nullable for ``column_names``.

    Prefer ``lineage`` when already built; otherwise derive from
    ``transformation_config`` via the pipeline plan-time lineage map.
    """
    lin = lineage
    if lin is None and transformation_config is not None:
        try:
            from .pipeline_no_null_report import build_pipeline_lineage_map

            lin = build_pipeline_lineage_map(transformation_config)
        except Exception:
            lin = {}
    return expand_nullable_via_lineage(
        lin,
        explicit=explicit,
        column_names=column_names,
    )


def mandatory_columns_for_step2(
    column_names: Sequence[str],
    *,
    nullable: frozenset[str] | None = None,
    transformation_config: dict[str, Any] | None = None,
    lineage: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    """Columns that must be non-NULL for a row to survive Step 2.

    ``nullable`` overrides the effective set when provided. Otherwise Explicit
    Nullable plus Inherited Nullable (from ``lineage`` /
    ``transformation_config``) are skipped.
    """
    if nullable is not None:
        skip = frozenset(nullable)
    elif transformation_config is not None or lineage is not None:
        skip = resolve_nullable_features(
            column_names,
            transformation_config=transformation_config,
            lineage=lineage,
        ).effective
    else:
        skip = _NULLABLE_SET
    out: list[str] = []
    for raw in column_names:
        name = str(raw or "").strip()
        if not name or name in skip:
            continue
        out.append(name)
    return out


def nullable_columns_present(
    column_names: Sequence[str],
    *,
    resolution: NullableResolution | None = None,
    transformation_config: dict[str, Any] | None = None,
    lineage: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    """Nullable columns present in ``column_names`` (explicit then inherited)."""
    if resolution is None and (transformation_config is not None or lineage is not None):
        resolution = resolve_nullable_features(
            column_names,
            transformation_config=transformation_config,
            lineage=lineage,
        )
    if resolution is not None:
        return resolution.present_all(column_names)
    names = {str(c).strip() for c in column_names if str(c).strip()}
    return [c for c in NULLABLE_FEATURE_LIST if c in names]


def format_nullable_classification(
    resolution: NullableResolution,
    *,
    column_names: Sequence[str] | None = None,
    max_inherited: int = 80,
) -> list[str]:
    """Human-readable Explicit / Inherited Nullable block with inherit traces."""
    if column_names is not None:
        explicit = resolution.present_explicit(column_names)
        inherited = resolution.present_inherited(column_names)
    else:
        explicit = [c for c in NULLABLE_FEATURE_LIST if c in resolution.explicit]
        inherited = sorted(resolution.inherited)

    lines = [
        "Nullable Classification",
        "",
        "Registry (Explicit)",
        "-" * 20,
    ]
    if explicit:
        for name in explicit:
            lines.append(f"✓ {name}")
    else:
        lines.append("(none)")

    lines.extend(
        [
            "",
            "Pipeline (Inherited)",
            "-" * 20,
        ]
    )
    if not inherited:
        lines.append("(none)")
    else:
        show = inherited[: max(0, int(max_inherited))]
        for name in show:
            lines.append(f"✓ {name}")
            parents = resolution.inheritance_parents(name)
            if parents:
                # Prefer a single primary parent when one; else list all.
                if len(parents) == 1:
                    lines.append(f"    ↳ inherits from {parents[0]}")
                else:
                    lines.append(
                        "    ↳ inherits from " + ", ".join(parents)
                    )
            else:
                lines.append("    ↳ inherits from (unknown parent)")
        if len(inherited) > len(show):
            lines.append(f"… and {len(inherited) - len(show)} more")
    return lines


__all__ = [
    "NULLABLE_FEATURE_LIST",
    "NullableResolution",
    "expand_nullable_via_lineage",
    "format_nullable_classification",
    "is_effectively_nullable",
    "is_nullable_feature",
    "mandatory_columns_for_step2",
    "nullable_columns_present",
    "nullable_feature_names",
    "resolve_nullable_features",
]
