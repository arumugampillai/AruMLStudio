"""Interaction transformation — pairwise arithmetic of existing columns.

Never materializes into Master / Feature Registry. Runs only in the Dataset
Builder pipeline after Lag / Difference / Return / Rolling so it can consume
both canonical and previously transformed columns.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .base import FeatureTransformation, TransformationResult, TransformContext
from .registry import register_transformation
from .time_shift import (
    LagConfigError,
    persist_sample_interval,
    resolve_sample_interval,
    resolve_transform_params,
)

# Canonical op id → auto-name infix
OP_SUFFIX: dict[str, str] = {
    "multiply": "_x_",
    "divide": "_div_",
    "add": "_plus_",
    "subtract": "_minus_",
    "min": "_min_",
    "max": "_max_",
    "absolute_difference": "_absdiff_",
}

# Aliases accepted in config → canonical id
OP_ALIASES: dict[str, str] = {
    "multiply": "multiply",
    "mul": "multiply",
    "x": "multiply",
    "*": "multiply",
    "divide": "divide",
    "div": "divide",
    "/": "divide",
    "add": "add",
    "plus": "add",
    "+": "add",
    "subtract": "subtract",
    "sub": "subtract",
    "minus": "subtract",
    "-": "subtract",
    "min": "min",
    "max": "max",
    "absolute_difference": "absolute_difference",
    "abs_diff": "absolute_difference",
    "absdiff": "absolute_difference",
}

_VALID_OPS = frozenset(OP_SUFFIX)
_DIV_ZERO_POLICIES = frozenset({"null", "zero", "fail"})

DEFAULT_SOFT_DEPENDS_ON: tuple[str, ...] = (
    "lag",
    "difference",
    "return",
    "rolling",
    "exponential_rolling",
    "ohlc_aggregation",
    "rolling_statistics",
    "rolling_ohlc",
)


def normalize_interaction_op(op: str | None) -> str:
    key = str(op or "multiply").strip().lower() or "multiply"
    canonical = OP_ALIASES.get(key)
    if canonical is None:
        raise LagConfigError(
            "Interaction Transformation\n"
            f"Invalid op={op!r}. Expected one of {sorted(_VALID_OPS)} "
            f"(aliases: {sorted(set(OP_ALIASES) - _VALID_OPS)})."
        )
    return canonical


def interaction_column_name(left: str, right: str, op: str = "multiply") -> str:
    """Deterministic output name: ``{left}{suffix}{right}``."""
    op_key = normalize_interaction_op(op)
    return f"{left}{OP_SUFFIX[op_key]}{right}"


def _apply_op(
    left: pd.Series,
    right: pd.Series,
    op: str,
    *,
    div_zero: str = "null",
    eps: float = 1e-12,
) -> pd.Series:
    if op == "multiply":
        return left * right
    if op == "divide":
        denom = right.astype(float)
        zero_mask = denom.abs() <= float(eps)
        if div_zero == "fail" and bool(zero_mask.any()):
            raise LagConfigError(
                "Interaction Transformation\n"
                "Divide-by-zero encountered (div_zero=fail)."
            )
        if div_zero == "zero":
            safe = denom.where(~zero_mask, np.nan)
            out = left.astype(float) / safe
            return out.where(~zero_mask, 0.0)
        # null (default): NaN where |right| <= eps
        safe = denom.where(~zero_mask, np.nan)
        return left.astype(float) / safe
    if op == "add":
        return left + right
    if op == "subtract":
        return left - right
    if op == "min":
        return pd.concat([left, right], axis=1).min(axis=1)
    if op == "max":
        return pd.concat([left, right], axis=1).max(axis=1)
    if op == "absolute_difference":
        return (left - right).abs()
    raise LagConfigError(
        "Interaction Transformation\n"
        f"Invalid op={op!r}. Expected one of {sorted(_VALID_OPS)}."
    )


def normalize_interaction_pair(pair: dict[str, Any]) -> dict[str, Any]:
    """Normalize one pair entry; raises LagConfigError on structural errors."""
    if not isinstance(pair, dict):
        raise LagConfigError(
            "Interaction Transformation\n"
            f"Invalid pair entry: {pair!r}"
        )
    left = str(pair.get("left") or "").strip()
    right = str(pair.get("right") or "").strip()
    if not left or not right:
        raise LagConfigError(
            "Interaction Transformation\n"
            "pair.left and pair.right are required."
        )
    op = normalize_interaction_op(pair.get("op"))
    output = str(pair.get("output") or "").strip()
    if not output:
        output = interaction_column_name(left, right, op)
    try:
        scale = float(pair.get("scale", 1.0))
    except (TypeError, ValueError):
        scale = 1.0
    div_zero = str(pair.get("div_zero") or "").strip().lower()
    eps_raw = pair.get("eps")
    out: dict[str, Any] = {
        "left": left,
        "right": right,
        "op": op,
        "output": output,
        "scale": scale,
    }
    if div_zero:
        if div_zero not in _DIV_ZERO_POLICIES:
            raise LagConfigError(
                "Interaction Transformation\n"
                f"Invalid div_zero={div_zero!r}. Expected one of {sorted(_DIV_ZERO_POLICIES)}."
            )
        out["div_zero"] = div_zero
    if eps_raw is not None and str(eps_raw).strip() != "":
        try:
            out["eps"] = float(eps_raw)
        except (TypeError, ValueError) as exc:
            raise LagConfigError(
                "Interaction Transformation\n"
                f"Invalid eps={eps_raw!r}."
            ) from exc
    return out


def validate_interaction_pairs(
    pairs: list[dict[str, Any]],
    *,
    existing_columns: set[str] | None = None,
    fail_on_duplicate_output: bool = True,
    allow_overwrite: bool = False,
    check_sources: bool = True,
) -> list[dict[str, Any]]:
    """Normalize + validate pairs; supports intra-step chaining."""
    if not pairs:
        raise LagConfigError(
            "Interaction Transformation\n"
            "params.pairs is empty.\n"
            "Provide [{left, right, op?, output?, scale?}]."
        )
    existing = set(existing_columns or ())
    seen_outputs: set[str] = set()
    available = set(existing)
    normalized: list[dict[str, Any]] = []
    for pair in pairs:
        norm = normalize_interaction_pair(pair)
        left, right, output = norm["left"], norm["right"], norm["output"]
        if check_sources and existing_columns is not None:
            missing = [c for c in (left, right) if c not in available]
            if missing:
                raise LagConfigError(
                    "Interaction Transformation\n"
                    "Feature not found\n"
                    + "\n".join(missing)
                )
        if fail_on_duplicate_output and output in seen_outputs:
            raise LagConfigError(
                "Interaction Transformation\n"
                f"Duplicate output name: {output}"
            )
        if not allow_overwrite and existing_columns is not None and output in existing:
            raise LagConfigError(
                "Interaction Transformation\n"
                f"Output already exists on frame: {output}"
            )
        seen_outputs.add(output)
        available.add(output)
        normalized.append(norm)
    return normalized


def interaction_lineage_node(
    left: str,
    right: str,
    op: str,
    output: str,
    *,
    left_lineage: dict[str, Any] | None = None,
    right_lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tree node describing how an interaction column was produced."""
    return {
        "feature": output,
        "transform": "interaction",
        "op": normalize_interaction_op(op),
        "left": left_lineage or {"feature": left},
        "right": right_lineage or {"feature": right},
    }


@register_transformation
class InteractionTransformation(FeatureTransformation):
    """``output = scale * (left op right)`` for configured column pairs."""

    id = "interaction"
    name = "Interaction"
    order = 50
    enabled = False
    # Soft guidance only — pipeline order (50) places Interaction after Lag/Diff/Return/Rolling.
    # Do not list hard depends_on; Interaction may run on Master columns alone.
    depends_on: list[str] = []
    params: dict[str, Any] = {}

    def describe(
        self,
        params: dict[str, Any] | None = None,
        *,
        upstream=None,
        master_features: list[str] | None = None,
        sample_interval_sec: float | int | None = None,
        enabled: bool | None = None,
    ):
        from .describe import (
            MASTER_STAGE_ID,
            OutputDescriptor,
            make_stage_descriptor,
        )

        del sample_interval_sec
        is_enabled = bool(self.enabled if enabled is None else enabled)
        params = dict(params or {})
        outputs: list[OutputDescriptor] = []
        input_sources = [MASTER_STAGE_ID]
        if upstream is not None:
            input_sources = [
                st.id
                for st in upstream.stages_before(self.id)
                if st.enabled or st.id == MASTER_STAGE_ID
            ] or [MASTER_STAGE_ID]
        try:
            for raw in list(params.get("pairs") or []):
                if not isinstance(raw, dict):
                    continue
                pair = normalize_interaction_pair(raw)
                outputs.append(
                    OutputDescriptor(
                        name=str(pair["output"]),
                        kind="interaction",
                        source_feature=str(pair.get("left") or ""),
                        op=str(pair.get("op") or "multiply"),
                        meta={
                            "left": pair.get("left"),
                            "right": pair.get("right"),
                            "scale": pair.get("scale", 1.0),
                        },
                    )
                )
        except Exception:
            outputs = []
        return make_stage_descriptor(
            self,
            enabled=is_enabled,
            outputs=outputs,
            input_sources=input_sources,
            notes=(
                "Pairwise arithmetic. May consume Master and any earlier "
                "transform outputs (order < 50)."
            ),
        )

    def transform(self, df: pd.DataFrame, context: TransformContext) -> TransformationResult:
        t0 = time.perf_counter()
        params = resolve_transform_params(self.id, getattr(self, "params", None), context.config)
        interval = resolve_sample_interval(context, params)
        pairs_raw = params.get("pairs") or []
        if not isinstance(pairs_raw, list) or not pairs_raw:
            raise LagConfigError(
                "Interaction Transformation\n"
                "params.pairs is empty.\n"
                "Provide [{left, right, op, output?, scale?}]."
            )
        persist_sample_interval(context, self.id, interval)
        if context.cancelled():
            return TransformationResult.passthrough(
                df,
                transformation_id=self.id,
                transformation_name=self.name,
            )

        div_zero_default = str(params.get("div_zero") or "null").strip().lower() or "null"
        if div_zero_default not in _DIV_ZERO_POLICIES:
            raise LagConfigError(
                "Interaction Transformation\n"
                f"Invalid params.div_zero={div_zero_default!r}."
            )
        try:
            eps_default = float(params.get("eps", 1e-12))
        except (TypeError, ValueError):
            eps_default = 1e-12
        fail_dup = bool(params.get("fail_on_duplicate_output", True))
        allow_overwrite = bool(params.get("overwrite", False))

        pairs = validate_interaction_pairs(
            [p for p in pairs_raw if p is not None],
            existing_columns=set(df.columns),
            fail_on_duplicate_output=fail_dup,
            allow_overwrite=allow_overwrite,
        )

        # Attach defaults so Polars path sees per-pair div_zero/eps.
        for pair in pairs:
            pair.setdefault("div_zero", div_zero_default)
            pair.setdefault("eps", eps_default)

        created = [str(p["output"]) for p in pairs]
        lineages = [
            interaction_lineage_node(p["left"], p["right"], p["op"], p["output"])
            for p in pairs
        ]
        total = len(pairs)

        def _pandas_fallback(frame: pd.DataFrame) -> pd.DataFrame:
            new_cols: dict[str, pd.Series] = {}

            def _series_for(name: str) -> pd.Series:
                if name in new_cols:
                    return pd.to_numeric(new_cols[name], errors="coerce")
                return pd.to_numeric(frame[name], errors="coerce")

            for pair in pairs:
                series = _apply_op(
                    _series_for(pair["left"]),
                    _series_for(pair["right"]),
                    pair["op"],
                    div_zero=str(pair.get("div_zero") or div_zero_default),
                    eps=float(pair.get("eps", eps_default)),
                )
                scale = float(pair.get("scale", 1.0))
                if scale != 1.0:
                    series = series * scale
                series.name = pair["output"]
                new_cols[pair["output"]] = series
            if new_cols:
                added = pd.DataFrame(new_cols, index=frame.index)
                overlap = [c for c in added.columns if c in frame.columns]
                base = frame.drop(columns=overlap, errors="ignore") if overlap else frame
                return pd.concat([base, added], axis=1)
            return frame

        from .polars_ops import apply_interaction_via_polars

        out = apply_interaction_via_polars(
            df,
            pairs=pairs,
            pandas_fallback=_pandas_fallback,
        )
        context.report_progress(f"Interaction: {total}/{total} columns", total, total)

        # Stash lineage for export metadata consumers.
        extras = context.extras if isinstance(context.extras, dict) else {}
        extras["interaction_lineage"] = lineages
        extras["interaction_summary"] = {
            "pair_count": len(created),
            "ops_used": sorted({p["op"] for p in pairs}),
            "created_columns": list(created),
        }
        context.extras = extras

        elapsed = round(max(time.perf_counter() - t0, 0.0), 4)
        context.log("Interaction Transformation")
        context.log(f"Columns Created : {len(created)}")
        context.log(f"Elapsed : {elapsed:.2f} s")
        return TransformationResult(
            frame=out,
            created_columns=created,
            elapsed_sec=elapsed,
            rows_processed=int(len(out)),
            transformation_id=self.id,
            transformation_name=self.name,
            messages=[
                f"Pairs : {len(created)}",
                f"Columns Created : {len(created)}",
                f"Ops : {', '.join(sorted({p['op'] for p in pairs}))}",
                "frame_backend=polars_interaction",
            ],
        )
