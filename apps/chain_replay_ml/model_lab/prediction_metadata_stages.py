"""Registry-driven pipeline-stage definitions for the Prediction Dataset Metadata tab.

Centralizes "which ``prediction_dataset`` columns does build stage X own" plus
optional human-readable notes explaining *why* a stage isn't 100% covered.
Both the SQL compute layer
(``prediction_dataset_metadata.compute_prediction_dataset_metadata``) and the
Tk UI (Research Lab → Prediction Dataset → Metadata) read this registry
instead of hardcoding the stage list independently — adding a future stage
(SHAP, Meta-model, ...) means registering one new ``StageSpec`` here, nothing
else changes.

Every column not explicitly claimed by a pipeline stage (``trading_day``,
``timestamp``, ``token``, ``master_row_id``, ``sf_*`` embedded features, ...)
falls into the catch-all ``Identity/Other`` stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from chain_replay_ml.training.prediction_packages import (
    PROBABILITY_LADDER,
    PROBABILITY_OUTPUT_COLUMNS,
)

from .prediction_schema import RR_HIT_COLUMNS
from .target_spec import ALL_TARGET_SPECS, inference_columns_for_key

STATUS_OK = "ok"
STATUS_PARTIAL = "partial"
STATUS_NONE = "none"
STATUS_NOT_BUILT = "not_built"

# Human-readable status text — never icon-only (screen readers / colorblind users).
STATUS_LABEL: dict[str, str] = {
    STATUS_OK: "\u2713 Complete",  # ✓ Complete
    STATUS_PARTIAL: "\u26a0 Partial",  # ⚠ Partial
    STATUS_NONE: "\u2717 Empty/Failed",  # ✗ Empty/Failed
    STATUS_NOT_BUILT: "Not Built",
}

# Coverage thresholds for stage/column status classification.
FULL_COVERAGE_PCT = 99.5
ZERO_COVERAGE_PCT = 0.01
# A column/stage counts as "ready" once it carries any signal at all — this is
# what makes "1 / 6 ladder models present" match user intent when only one
# classifier member has been trained so far.
READY_COVERAGE_PCT = 0.0

# Coverage → color bucket, shared by Column Coverage and Stage Coverage so the
# same row always renders with the same traffic-light color in both tables.
BUCKET_FULL = "full"       # 🟢 ~100%
BUCKET_HIGH = "high"       # 🟡 partial, high
BUCKET_LOW = "low"         # 🟠 partial, low
BUCKET_EMPTY = "empty"     # 🔴 0% / no data

COVERAGE_BUCKET_EMOJI: dict[str, str] = {
    BUCKET_FULL: "\U0001F7E2",   # 🟢
    BUCKET_HIGH: "\U0001F7E1",   # 🟡
    BUCKET_LOW: "\U0001F7E0",    # 🟠
    BUCKET_EMPTY: "\U0001F534",  # 🔴
}

# Matches the Research Lab's existing OK/WARN/MUTED palette so new coloring
# blends into the rest of the Tk UI instead of introducing a new scheme.
COVERAGE_BUCKET_COLOR: dict[str, str] = {
    BUCKET_FULL: "#2E7D32",
    BUCKET_HIGH: "#B8860B",
    BUCKET_LOW: "#C77800",
    BUCKET_EMPTY: "#C62828",
}

REGRESSION_ANCHOR = "predicted_future_ltp"

# Confidence inference writes one <target>_pred column per TargetSpec.
CONFIDENCE_PRED_COLUMNS: tuple[str, ...] = tuple(
    inference_columns_for_key(t.key)["pred"] for t in ALL_TARGET_SPECS
)
_CONFIDENCE_LABEL_BY_COLUMN: dict[str, str] = {
    inference_columns_for_key(t.key)["pred"]: t.label for t in ALL_TARGET_SPECS
}

TRIPLE_BARRIER_COLUMNS: tuple[str, ...] = ("tb_pred_probability", "tb_pred_class")

COMPUTE_OUTCOME_COLUMNS: tuple[str, ...] = (
    "maximum_profit",
    "maximum_drawdown",
    "dd_before_target",
    "time_to_max_profit",
    "time_to_max_drawdown",
    "time_to_dd_before_target",
    "time_to_target",
    "target_reached",
    "target_reached_at",
    "max_profit_at",
    "max_drawdown_at",
    "exit_at",
    *RR_HIT_COLUMNS,
)

_LADDER_LABEL_BY_COLUMN: dict[str, str] = {
    str(item["output_column"]): str(item["label"]) for item in PROBABILITY_LADDER
}


def coverage_bucket(coverage_pct: float, *, populated: int | None = None) -> str:
    """Traffic-light bucket for one coverage percentage.

    ``populated`` (when known) takes priority over the rounded percentage so
    a truly-zero column never renders as anything but ``empty``.
    """
    if populated is not None and populated <= 0:
        return BUCKET_EMPTY
    pct = float(coverage_pct or 0.0)
    if pct <= ZERO_COVERAGE_PCT:
        return BUCKET_EMPTY
    if pct >= FULL_COVERAGE_PCT:
        return BUCKET_FULL
    if pct >= 50.0:
        return BUCKET_HIGH
    return BUCKET_LOW


def coverage_emoji(coverage_pct: float, *, populated: int | None = None) -> str:
    return COVERAGE_BUCKET_EMOJI[coverage_bucket(coverage_pct, populated=populated)]


def stage_status(coverage_pct: float, *, has_columns: bool, total_rows: int) -> str:
    if total_rows <= 0 or not has_columns:
        return STATUS_NOT_BUILT
    if coverage_pct <= ZERO_COVERAGE_PCT:
        return STATUS_NONE
    if coverage_pct >= FULL_COVERAGE_PCT:
        return STATUS_OK
    return STATUS_PARTIAL


@dataclass(frozen=True)
class StageContext:
    """Everything a stage's notes builder needs — no DB access required."""

    status: str
    coverage_pct: float
    expected_columns: tuple[str, ...]
    stats_by_name: dict[str, dict[str, Any]]
    total_rows: int
    package_members: tuple[dict[str, Any], ...] | None = None


NotesBuilder = Callable[[StageContext], str]
ColumnResolver = Callable[[Sequence[str]], tuple[str, ...]]


@dataclass(frozen=True)
class StageSpec:
    """One pipeline stage's identity + the columns it owns."""

    key: str
    label: str
    resolve_columns: ColumnResolver
    notes: NotesBuilder | None = None
    # True only for the Identity/Other bucket — its columns are resolved as
    # "everything no other stage claimed", not via ``resolve_columns``.
    catch_all: bool = False


def _regression_columns(all_columns: Sequence[str]) -> tuple[str, ...]:
    """``predicted_future_ltp`` plus any other ``predicted_*`` columns present."""
    present = set(all_columns)
    dynamic = sorted(
        c for c in present if c.startswith("predicted_") and c != REGRESSION_ANCHOR
    )
    if REGRESSION_ANCHOR in present:
        return (REGRESSION_ANCHOR, *dynamic)
    return tuple(dynamic)


def _ladder_columns(all_columns: Sequence[str]) -> tuple[str, ...]:
    present = set(all_columns)
    return tuple(c for c in PROBABILITY_OUTPUT_COLUMNS if c in present)


def _triple_barrier_columns(all_columns: Sequence[str]) -> tuple[str, ...]:
    present = set(all_columns)
    return tuple(c for c in TRIPLE_BARRIER_COLUMNS if c in present)


def _compute_outcome_columns(all_columns: Sequence[str]) -> tuple[str, ...]:
    present = set(all_columns)
    return tuple(c for c in COMPUTE_OUTCOME_COLUMNS if c in present)


def _confidence_columns(all_columns: Sequence[str]) -> tuple[str, ...]:
    present = set(all_columns)
    return tuple(c for c in CONFIDENCE_PRED_COLUMNS if c in present)


def _no_columns(_all_columns: Sequence[str]) -> tuple[str, ...]:
    return ()


def _populated_rows(ctx: StageContext, columns: Sequence[str]) -> int:
    best = 0
    for c in columns:
        stat = ctx.stats_by_name.get(c) or {}
        best = max(best, int(stat.get("populated") or 0))
    return best


def _generic_notes(stage_label: str) -> NotesBuilder:
    """Fallback notes for stages with no member-level breakdown available."""

    def _build(ctx: StageContext) -> str:
        if ctx.status in (STATUS_OK, STATUS_NOT_BUILT):
            return ""
        if not ctx.expected_columns:
            return f"{stage_label} columns are not present in this schema version."
        if ctx.status == STATUS_NONE:
            return f"{stage_label} not scored for any of {ctx.total_rows:,} rows."
        populated_rows = _populated_rows(ctx, ctx.expected_columns)
        return (
            f"{stage_label} partially built — {populated_rows:,} of "
            f"{ctx.total_rows:,} rows scored ({ctx.coverage_pct:.1f}% avg column coverage)."
        )

    return _build


def _confidence_notes(ctx: StageContext) -> str:
    if ctx.status in (STATUS_OK, STATUS_NOT_BUILT):
        return ""
    if ctx.status == STATUS_NONE:
        return "No confidence models scored for any rows yet — not built."
    missing = [
        _CONFIDENCE_LABEL_BY_COLUMN.get(c, c)
        for c in ctx.expected_columns
        if float((ctx.stats_by_name.get(c) or {}).get("coverage_pct") or 0.0)
        <= ZERO_COVERAGE_PCT
    ]
    if missing:
        shown = ", ".join(missing[:6])
        more = f" +{len(missing) - 6} more" if len(missing) > 6 else ""
        return f"Not scored yet (0% populated): {shown}{more}."
    return (
        f"Partial coverage ({ctx.coverage_pct:.1f}% avg) — some confidence "
        "targets are scored on fewer rows than others (partial days)."
    )


def _ladder_notes(ctx: StageContext) -> str:
    if ctx.status == STATUS_NOT_BUILT:
        return ""
    if ctx.package_members:
        missing_labels = [
            str(m.get("label") or m.get("key"))
            for m in ctx.package_members
            if not m.get("available")
        ]
        if not missing_labels:
            if ctx.status == STATUS_OK:
                return ""
            return "All ladder members are registered, but some rows are not yet scored."
        return f"Missing ladder models: {', '.join(missing_labels)}."

    # No live package context (pure DB-only compute) — infer availability from
    # column population: a ladder member with 0% coverage almost always means
    # that classifier was never trained/selected for this package.
    missing = [
        _LADDER_LABEL_BY_COLUMN.get(c, c)
        for c in ctx.expected_columns
        if float((ctx.stats_by_name.get(c) or {}).get("coverage_pct") or 0.0)
        <= ZERO_COVERAGE_PCT
    ]
    if ctx.status == STATUS_NONE:
        return "No ladder models scored for any rows yet — not built."
    if missing:
        return f"Missing ladder models (inferred — 0% populated): {', '.join(missing)}."
    return f"Partial coverage ({ctx.coverage_pct:.1f}% avg) across ladder members."


def _identity_notes(_ctx: StageContext) -> str:
    return ""


STAGE_REGISTRY: tuple[StageSpec, ...] = (
    StageSpec(
        key="regression",
        label="Regression",
        resolve_columns=_regression_columns,
        notes=_generic_notes("Regression"),
    ),
    StageSpec(
        key="probability_ladder",
        label="Probability Ladder",
        resolve_columns=_ladder_columns,
        notes=_ladder_notes,
    ),
    StageSpec(
        key="triple_barrier",
        label="Triple Barrier",
        resolve_columns=_triple_barrier_columns,
        notes=_generic_notes("Triple Barrier scorer"),
    ),
    StageSpec(
        key="compute_outcomes",
        label="Compute Outcomes",
        resolve_columns=_compute_outcome_columns,
        notes=_generic_notes("Compute Outcomes"),
    ),
    StageSpec(
        key="confidence",
        label="Confidence",
        resolve_columns=_confidence_columns,
        notes=_confidence_notes,
    ),
    StageSpec(
        key="identity",
        label="Identity/Other",
        resolve_columns=_no_columns,
        notes=_identity_notes,
        catch_all=True,
    ),
)

STAGE_ORDER: tuple[str, ...] = tuple(spec.label for spec in STAGE_REGISTRY)
STAGE_BY_KEY: dict[str, StageSpec] = {spec.key: spec for spec in STAGE_REGISTRY}
IDENTITY_STAGE_LABEL: str = next(spec.label for spec in STAGE_REGISTRY if spec.catch_all)


def assign_column_stages(all_columns: Sequence[str]) -> dict[str, str]:
    """column name -> owning stage label, every physical column accounted for."""
    assigned: dict[str, str] = {}
    for spec in STAGE_REGISTRY:
        if spec.catch_all:
            continue
        for col in spec.resolve_columns(all_columns):
            assigned.setdefault(col, spec.label)
    for col in all_columns:
        assigned.setdefault(col, IDENTITY_STAGE_LABEL)
    return assigned


def stage_expected_columns(
    spec: StageSpec, all_columns: Sequence[str], assigned: dict[str, str]
) -> tuple[str, ...]:
    """Columns a stage owns, in physical column order."""
    if spec.catch_all:
        return tuple(c for c in all_columns if assigned.get(c) == spec.label)
    return spec.resolve_columns(all_columns)
