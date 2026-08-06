"""Pipeline No-Null diagnostics  -  attribution for Analysis / Feature Transformation.

Diagnostics only: never modifies frames, parquet, or registry.

Important semantics:
- The transformation pipeline adds columns; it does **not** drop rows.
- A later No-Null Step 2 *would* drop rows that have NULL in mandatory
  pipeline columns  -  that is a separate policy step.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# Feature primary class (actionability)
CLASS_INHERITED = "inherited"  # Class 1  -  Registry/parent propagation
CLASS_PIPELINE = "pipeline_created"  # Class 2  -  genuine new pipeline NULLs
CLASS_MATH = "mathematical"  # Class 3  -  div-by-zero etc.
CLASS_WARMUP = "warmup"  # Expected time-shift / rolling warm-up
CLASS_UNEXPECTED = "unexpected"  # Mixed / unclassified remainder

_CLASS_LABEL = {
    CLASS_INHERITED: "Inherited from Registry",
    CLASS_PIPELINE: "Pipeline-created (NEW)",
    CLASS_MATH: "Mathematical",
    CLASS_WARMUP: "Warm-up (expected)",
    CLASS_UNEXPECTED: "Unexpected / mixed",
}

# Unique-row priority when a row has several NULL causes (higher = more actionable)
_CLASS_PRIORITY = {
    CLASS_PIPELINE: 4,
    CLASS_MATH: 3,
    CLASS_UNEXPECTED: 2,
    CLASS_WARMUP: 1,
    CLASS_INHERITED: 0,
}


def build_pipeline_lineage_map(
    transformation_config: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Map pipeline output column → transform / parents / op (plan-time).

    Walks **every** config entry (including duplicate transform ids such as
    multiple ``rolling_statistics`` stages) so z-score / packaging outputs
    keep their source features.
    """
    if not isinstance(transformation_config, dict):
        return {}

    out: dict[str, dict[str, Any]] = {}

    def _put(
        name: str,
        *,
        transform: str,
        parents: list[str],
        kind: str = "",
        op: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        n = str(name or "").strip()
        if not n:
            return
        uniq: list[str] = []
        seen: set[str] = set()
        for p in parents:
            ps = str(p or "").strip()
            if ps and ps not in seen:
                seen.add(ps)
                uniq.append(ps)
        out[n] = {
            "transform": transform,
            "transform_name": transform,
            "kind": kind or transform,
            "op": op,
            "parents": uniq,
            "meta": dict(meta or {}),
        }

    # 1) Raw config entries (handles duplicate ids)
    for entry in list(transformation_config.get("transformations") or []):
        if not isinstance(entry, dict) or not entry.get("enabled", False):
            continue
        tid = str(entry.get("id") or "").strip()
        params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
        if tid == "interaction":
            for raw in list(params.get("pairs") or []):
                if not isinstance(raw, dict):
                    continue
                left = str(raw.get("left") or "").strip()
                right = str(raw.get("right") or "").strip()
                op = str(raw.get("op") or "multiply").strip()
                output = str(raw.get("output") or "").strip()
                if not output and left and right:
                    try:
                        from .transformations.interaction import interaction_column_name

                        output = interaction_column_name(left, right, op)
                    except Exception:
                        continue
                if output:
                    _put(
                        output,
                        transform="interaction",
                        parents=[left, right],
                        kind="interaction",
                        op=op,
                        meta={"left": left, "right": right},
                    )
        elif tid == "rolling_statistics":
            feats = [str(f).strip() for f in (params.get("features") or []) if str(f).strip()]
            stat = str(params.get("stat") or "zscore").strip().lower() or "zscore"
            for feat in feats:
                for w in list(params.get("windows") or []):
                    if isinstance(w, dict):
                        col = str(w.get("column") or "").strip()
                    else:
                        col = ""
                    if col:
                        _put(
                            col,
                            transform="rolling_statistics",
                            parents=[feat],
                            kind="rolling_statistics",
                            op=stat,
                            meta={"feature": feat, "stat": stat},
                        )
        elif tid in {"lag", "difference", "return", "difference_clip", "anchor_return"}:
            feats = [str(f).strip() for f in (params.get("features") or []) if str(f).strip()]
            horizons = list(params.get("horizons") or params.get("windows") or [])
            for feat in feats:
                for h in horizons:
                    if not isinstance(h, dict):
                        continue
                    col = str(h.get("column") or "").strip()
                    if col:
                        _put(
                            col,
                            transform=tid,
                            parents=[feat],
                            kind=tid,
                            op=tid,
                            meta={"feature": feat},
                        )
        elif tid in {"rolling", "exponential_rolling", "ohlc_aggregation", "math", "normalization"}:
            feats = [str(f).strip() for f in (params.get("features") or []) if str(f).strip()]
            for feat in feats:
                for key in ("windows", "horizons", "periods", "ops", "outputs"):
                    for item in list(params.get(key) or []):
                        if isinstance(item, dict):
                            col = str(item.get("column") or item.get("output") or "").strip()
                            if col:
                                _put(
                                    col,
                                    transform=tid,
                                    parents=[feat],
                                    kind=tid,
                                    meta={"feature": feat},
                                )

    # 2) Plan-time describe() for anything still missing
    try:
        from .transformations.describe import MASTER_STAGE_ID, describe_pipeline_stages

        plan = describe_pipeline_stages(transformation_config)
        for st in plan.stages:
            if str(getattr(st, "id", "") or "") == MASTER_STAGE_ID:
                continue
            if not bool(getattr(st, "enabled", False)):
                continue
            for d in getattr(st, "output_descriptors", ()) or ():
                name = str(getattr(d, "name", "") or "").strip()
                if not name or name in out:
                    continue
                meta = dict(getattr(d, "meta", None) or {})
                parents: list[str] = []
                for key in ("left", "right", "feature", "source"):
                    val = meta.get(key)
                    if val is not None and str(val).strip():
                        parents.append(str(val).strip())
                src = str(getattr(d, "source_feature", "") or "").strip()
                if src and src not in parents:
                    parents.insert(0, src)
                _put(
                    name,
                    transform=str(getattr(st, "id", "") or ""),
                    parents=parents,
                    kind=str(getattr(d, "kind", "") or getattr(st, "id", "") or ""),
                    op=str(getattr(d, "op", "") or ""),
                    meta=meta,
                )
    except Exception:
        pass

    return out


# Heuristic: names that look like pipeline outputs (not Registry leaves).
_PIPELINE_NAME_HINTS = (
    "_lag_",
    "_diff_",
    "_return_",
    "_change_",
    "_x_",
    "_div_",
    "_plus_",
    "_minus_",
    "_roll_",
    "_zscore_",
    "zscore_",
    "_slope_",
    "_clip_",
    "_norm_",
    "_pct_",
)


def _looks_pipeline_produced(name: str) -> bool:
    n = str(name or "").strip().lower()
    if not n:
        return False
    return any(tok in n for tok in _PIPELINE_NAME_HINTS)


def validate_pipeline_lineage_parents(
    lineage: Mapping[str, Mapping[str, Any]] | None,
    *,
    known_columns: Sequence[str] | None = None,
) -> list[str]:
    """Warn when a pipeline feature depends on a missing dependency-graph node.

    A parent is resolved if it is:
    - a pipeline output (key in ``lineage``), or
    - listed in ``known_columns`` (Registry / frame inputs), when provided.

    Without ``known_columns``, Registry-like leaves are assumed OK; only parents
    that *look* pipeline-produced but are absent from ``lineage`` warn — those
    usually mean a missing / disabled upstream transform.
    """
    if not lineage:
        return []

    graph: set[str] = {str(k).strip() for k in lineage.keys() if str(k).strip()}
    if known_columns is not None:
        graph |= {str(c).strip() for c in known_columns if str(c).strip()}

    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()
    for raw_child, info in lineage.items():
        child = str(raw_child or "").strip()
        if not child or not isinstance(info, Mapping):
            continue
        for raw_parent in list(info.get("parents") or []):
            parent = str(raw_parent or "").strip()
            if not parent or parent in graph:
                continue
            key = (child, parent)
            if key in seen:
                continue
            # Config-only mode: skip plain Registry leaves (no pipeline naming).
            if known_columns is None and not _looks_pipeline_produced(parent):
                continue
            seen.add(key)
            warnings.append(
                f"Lineage gap: {child!r} depends on {parent!r}, which is not "
                f"present in the dependency graph "
                f"(missing upstream transform or unknown feature?)"
            )
    warnings.sort()
    return warnings


def _pct(part: int, whole: int) -> str:
    if whole <= 0:
        return "0.0%"
    return f"{100.0 * part / whole:.1f}%"


def _is_warmup_name(feature: str, kind: str = "") -> bool:
    f = str(feature or "")
    k = str(kind or "").lower()
    if any(tok in f for tok in ("_lag_", "_diff_", "_return_", "_change_")):
        return True
    if k in {"lag", "difference", "return", "difference_clip", "anchor_return"}:
        return True
    # Rolling z-score / windows: warm-up until window fills  -  expected design
    if k in {"rolling_statistics", "rolling", "exponential_rolling", "ohlc_aggregation"}:
        return True
    if "zscore" in f or "_slope_" in f:
        return True
    return False


def _classify_feature_nulls(
    frame: Any,
    feature: str,
    *,
    lineage: dict[str, Any] | None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Break down why ``feature`` is NULL; assign Class 1/2/3 primary label."""
    series = frame[feature]
    null_mask = series.isna().to_numpy()
    n_null = int(null_mask.sum())
    result: dict[str, Any] = {
        "null_rows": n_null,
        "propagated_any_parent": 0,
        "propagated_all_parents": 0,
        "new_null_parents_present": 0,
        "divide_by_zero": 0,
        "expected": False,
        "null_class": CLASS_UNEXPECTED,
        "reason": "Unknown",
        "action": "",
    }
    if n_null <= 0:
        result["reason"] = "No NULLs"
        result["null_class"] = CLASS_INHERITED
        return result

    lin = lineage or {}
    parents = [p for p in (lin.get("parents") or []) if p in frame.columns]
    kind = str(lin.get("kind") or lin.get("transform") or "").lower()
    op = str(lin.get("op") or "").lower()
    warmup_like = _is_warmup_name(feature, kind)

    if not parents:
        if warmup_like:
            result["reason"] = "Warm-up / window fill (expected pipeline design)"
            result["expected"] = True
            result["null_class"] = CLASS_WARMUP
            result["action"] = "Not a bug  -  rolling/time-shift warm-up."
            result["new_null_parents_present"] = n_null
        else:
            result["reason"] = "New NULL (no parent lineage in config)"
            result["expected"] = False
            result["null_class"] = CLASS_PIPELINE
            result["action"] = "Investigate  -  pipeline produced NULL without mapped parents."
            result["new_null_parents_present"] = n_null
        return result

    parent_nulls = [frame[p].isna().to_numpy() for p in parents]
    any_parent = parent_nulls[0].copy()
    all_parents = parent_nulls[0].copy()
    for m in parent_nulls[1:]:
        any_parent |= m
        all_parents &= m

    prop_any = int((null_mask & any_parent).sum())
    prop_all = int((null_mask & all_parents).sum()) if len(parents) > 1 else prop_any
    parents_ok = ~any_parent
    new_null = int((null_mask & parents_ok).sum())
    result["propagated_any_parent"] = prop_any
    result["propagated_all_parents"] = prop_all
    result["new_null_parents_present"] = new_null

    div_zero = 0
    if op in {"divide", "div", "/"} and len(parents) >= 2:
        left, right = parents[0], parents[1]
        try:
            r = frame[right].astype(float)
            l_ok = frame[left].notna().to_numpy()
            r_ok = frame[right].notna().to_numpy()
            z = (r.abs() <= float(eps)).to_numpy()
            div_zero = int((null_mask & l_ok & r_ok & z).sum())
        except Exception:
            div_zero = 0
    # Also detect /_div_/ naming when op missing
    if div_zero <= 0 and "_div_" in feature and len(parents) >= 2:
        left, right = parents[0], parents[1]
        try:
            r = frame[right].astype(float)
            l_ok = frame[left].notna().to_numpy()
            r_ok = frame[right].notna().to_numpy()
            z = (r.abs() <= float(eps)).to_numpy()
            div_zero = int((null_mask & l_ok & r_ok & z).sum())
        except Exception:
            div_zero = 0
    result["divide_by_zero"] = div_zero

    # Class assignment (feature primary)
    if prop_any >= n_null * 0.9:
        result["reason"] = (
            "Inherited from Registry  -  no new NULL introduced"
            if len(parents) == 1 or prop_all < n_null * 0.5
            else "Inherited from Registry  -  multiple parents NULL together"
        )
        result["expected"] = True
        result["null_class"] = CLASS_INHERITED
        result["action"] = "Interaction/pipeline is innocent; fix or accept upstream."
    elif div_zero > 0 and div_zero >= max(new_null - div_zero, 1) * 0.5:
        result["reason"] = f"Mathematical  -  divide-by-zero ({div_zero:,} rows)"
        result["expected"] = True
        result["null_class"] = CLASS_MATH
        result["action"] = "Not a Registry/pipeline bug  -  denom ~ 0."
    elif new_null >= n_null * 0.9 and warmup_like:
        result["reason"] = "Warm-up / window fill (parents present)"
        result["expected"] = True
        result["null_class"] = CLASS_WARMUP
        result["action"] = "Expected until rolling/time window fills."
    elif new_null >= n_null * 0.9:
        result["reason"] = "NEW NULL introduced by pipeline (parents present)"
        result["expected"] = False
        result["null_class"] = CLASS_PIPELINE
        result["action"] = "Highest priority  -  inspect formula / stage."
    else:
        bits = []
        if prop_any:
            bits.append(f"inherited={prop_any:,}")
        if new_null:
            bits.append(f"new={new_null:,}")
        if div_zero:
            bits.append(f"math_div0={div_zero:,}")
        result["reason"] = "Mixed (" + ", ".join(bits) + ")" if bits else "Unknown"
        # Prefer actionable class for mixed
        if new_null > prop_any and not warmup_like:
            result["null_class"] = CLASS_PIPELINE
            result["expected"] = False
            result["action"] = "Mixed  -  prioritize NEW component."
        elif div_zero > 0:
            result["null_class"] = CLASS_MATH
            result["expected"] = True
            result["action"] = "Mixed  -  includes mathematical NULLs."
        elif prop_any > 0:
            result["null_class"] = CLASS_INHERITED
            result["expected"] = True
            result["action"] = "Mostly inherited; NEW portion secondary."
        else:
            result["null_class"] = CLASS_UNEXPECTED
            result["expected"] = False
            result["action"] = "Unclassified  -  inspect dependency chain."

    return result


def build_pipeline_no_null_report_text(
    frame: Any,
    *,
    pipeline_columns: Sequence[str],
    transformation_config: dict[str, Any] | None = None,
    rows_after_filter: int | None = None,
    filter_applied: bool = False,
    top_n: int = 25,
    chain_examples: int = 8,
) -> str:
    """Format a Pipeline No-Null Report for Activity Log / diagnostics."""
    import numpy as np

    if frame is None or not hasattr(frame, "columns"):
        return "Pipeline No-Null Report: no frame available"

    n_in = int(len(frame))
    pipe_cols = [str(c) for c in pipeline_columns if str(c) in frame.columns]
    if not pipe_cols:
        pipe_cols = [
            str(c)
            for c in frame.columns
            if any(
                tok in str(c)
                for tok in (
                    "_lag_",
                    "_diff_",
                    "_return_",
                    "_x_",
                    "_div_",
                    "_plus_",
                    "_minus_",
                    "_roll_",
                    "zscore",
                    "_slope_",
                )
            )
        ]

    lineage = build_pipeline_lineage_map(transformation_config)
    from .nullable_features import (
        expand_nullable_via_lineage,
        format_nullable_classification,
        mandatory_columns_for_step2,
    )

    # Seed inheritance with frame columns + pipeline outputs so Registry parents
    # (e.g. current_iv) mark downstream pipeline cols as Inherited Nullable.
    nullable_scope = list(
        dict.fromkeys(
            [str(c) for c in frame.columns]
            + [str(c) for c in pipe_cols]
        )
    )
    nullable_res = expand_nullable_via_lineage(
        lineage,
        column_names=nullable_scope,
    )
    inherited_nullable = set(nullable_res.inherited)
    lineage_gap_warnings = validate_pipeline_lineage_parents(
        lineage,
        known_columns=nullable_scope,
    )

    n_out_filter = int(rows_after_filter) if rows_after_filter is not None else n_in
    removed_filter = (
        max(n_in - n_out_filter, 0)
        if filter_applied or rows_after_filter is not None
        else 0
    )

    present = [c for c in pipe_cols if c in frame.columns]
    # Step-2 candidates exclude Explicit + Inherited Nullable (same as filter).
    step2_mandatory = mandatory_columns_for_step2(
        present,
        nullable=nullable_res.effective,
    )
    step2_idx = [present.index(c) for c in step2_mandatory if c in present]
    if present:
        mat = np.column_stack([frame[c].notna().to_numpy() for c in present])
        if step2_idx:
            complete = mat[:, step2_idx].all(axis=1)
        else:
            complete = np.ones(n_in, dtype=bool)
        would_keep = int(complete.sum())
        would_remove = n_in - would_keep
        incomplete = ~complete
    else:
        mat = None
        complete = np.ones(n_in, dtype=bool)
        incomplete = np.zeros(n_in, dtype=bool)
        would_keep = n_in
        would_remove = 0

    feature_rows: list[dict[str, Any]] = []
    # Unique-row class priority (only among incomplete rows)
    row_priority = np.full(n_in, -1, dtype=np.int8)

    for i, feat in enumerate(present):
        null_mask = (~mat[:, i]) if mat is not None else frame[feat].isna().to_numpy()
        n_null = int(null_mask.sum())
        if n_null <= 0:
            continue
        exclusive = 0
        if mat is not None and step2_idx and len(step2_idx) > 1 and i in step2_idx:
            others = [j for j in step2_idx if j != i]
            others_ok = mat[:, others].all(axis=1)
            exclusive = int((null_mask & others_ok).sum())
        elif mat is not None and step2_idx == [i]:
            exclusive = n_null

        lin = lineage.get(feat) or {}
        classified = _classify_feature_nulls(frame, feat, lineage=lin)
        # Dependency-graph Inherited Nullable overrides Class-2 / mixed:
        # these are by design downstream of Explicit Nullable Registry features.
        if feat in inherited_nullable:
            classified = dict(classified)
            parents = nullable_res.inheritance_parents(feat)
            parent_note = (
                ", ".join(parents) if parents else "nullable parent(s)"
            )
            classified["null_class"] = CLASS_INHERITED
            classified["expected"] = True
            classified["reason"] = (
                f"Inherited Nullable  -  inherits from {parent_note}"
            )
            classified["action"] = (
                "Not a pipeline bug; excluded from Step-2 No-Null automatically."
            )
            classified["inherits_from"] = list(parents)
        null_class = str(classified.get("null_class") or CLASS_UNEXPECTED)
        feature_rows.append(
            {
                "feature": feat,
                "null_rows": n_null,
                "exclusive": exclusive,
                "lineage": lin,
                "class": classified,
                "null_class": null_class,
                "new_null": int(classified.get("new_null_parents_present") or 0),
                "inherited": int(classified.get("propagated_any_parent") or 0),
                "math": int(classified.get("divide_by_zero") or 0),
                "inherited_nullable": feat in inherited_nullable,
            }
        )

        # Skip Class-2 row scoring for Inherited Nullable columns.
        if feat in inherited_nullable:
            score_inh = _CLASS_PRIORITY[CLASS_INHERITED]
            row_priority = np.maximum(
                row_priority, np.where(null_mask & incomplete, score_inh, row_priority)
            )
            continue

        # Update unique-row priority from this feature's component masks
        parents = [p for p in (lin.get("parents") or []) if p in frame.columns]
        score_pipeline = _CLASS_PRIORITY[CLASS_PIPELINE]
        score_math = _CLASS_PRIORITY[CLASS_MATH]
        score_warm = _CLASS_PRIORITY[CLASS_WARMUP]
        score_inh = _CLASS_PRIORITY[CLASS_INHERITED]

        if parents:
            any_parent = frame[parents[0]].isna().to_numpy()
            for p in parents[1:]:
                any_parent |= frame[p].isna().to_numpy()
            prop_m = null_mask & any_parent
            new_m = null_mask & ~any_parent
            # math subset of new
            math_m = np.zeros(n_in, dtype=bool)
            if classified.get("divide_by_zero"):
                op = str(lin.get("op") or "").lower()
                if (op in {"divide", "div", "/"} or "_div_" in feat) and len(parents) >= 2:
                    try:
                        r = frame[parents[1]].astype(float)
                        math_m = (
                            null_mask
                            & frame[parents[0]].notna().to_numpy()
                            & frame[parents[1]].notna().to_numpy()
                            & (r.abs() <= 1e-12).to_numpy()
                        )
                        new_m = new_m & ~math_m
                    except Exception:
                        pass
            if _is_warmup_name(feat, str(lin.get("kind") or "")):
                row_priority = np.maximum(
                    row_priority, np.where(new_m, score_warm, row_priority)
                )
            else:
                row_priority = np.maximum(
                    row_priority, np.where(new_m, score_pipeline, row_priority)
                )
            row_priority = np.maximum(
                row_priority, np.where(math_m, score_math, row_priority)
            )
            row_priority = np.maximum(
                row_priority, np.where(prop_m, score_inh, row_priority)
            )
        else:
            score = (
                score_warm
                if _is_warmup_name(feat, str(lin.get("kind") or ""))
                else score_pipeline
            )
            row_priority = np.maximum(
                row_priority, np.where(null_mask, score, row_priority)
            )

    # Restrict unique-row summary to incomplete (would-be-removed) rows
    row_priority = np.where(incomplete, row_priority, -1)
    summary_rows = {
        CLASS_INHERITED: int((row_priority == _CLASS_PRIORITY[CLASS_INHERITED]).sum()),
        CLASS_PIPELINE: int((row_priority == _CLASS_PRIORITY[CLASS_PIPELINE]).sum()),
        CLASS_MATH: int((row_priority == _CLASS_PRIORITY[CLASS_MATH]).sum()),
        CLASS_WARMUP: int((row_priority == _CLASS_PRIORITY[CLASS_WARMUP]).sum()),
        CLASS_UNEXPECTED: int(
            (
                (row_priority == _CLASS_PRIORITY[CLASS_UNEXPECTED])
                | ((row_priority < 0) & incomplete)
            ).sum()
        ),
    }
    # Fix unexpected: incomplete with priority still -1
    summary_rows[CLASS_UNEXPECTED] = int(((row_priority < 0) & incomplete).sum())

    lines: list[str] = [
        "=" * 72,
        "Pipeline No-Null Report",
        "=" * 72,
        "(diagnostics only  -  does not change the dataset)",
        "",
        "-" * 72,
        "Pipeline Stage (transforms only)",
        "-" * 72,
        f"Input rows                 {n_in:,}",
        f"Output rows                {n_in:,}",
        "Rows physically removed    0",
        "",
        "The pipeline adds columns; it does not delete rows.",
        "",
        "-" * 72,
        "If Step-2 No-Null applied on pipeline columns",
        "-" * 72,
    ]
    if filter_applied:
        lines.extend(
            [
                f"Rows before filter         {n_in:,}",
                f"Rows after filter          {n_out_filter:,}",
                f"Rows removed by No-Null    {removed_filter:,} "
                f"({_pct(removed_filter, n_in)})",
            ]
        )
    else:
        lines.extend(
            [
                f"Would remove               {would_remove:,} "
                f"({_pct(would_remove, n_in)})",
                f"Would keep                 {would_keep:,}",
                "(No null data was OFF  -  hypothetical Step-2 impact only)",
            ]
        )

    lines.extend(
        [
            "",
            f"Pipeline columns analysed: {len(present)}",
            f"Pipeline columns with NULLs: {len(feature_rows)}",
            f"Step-2 mandatory (excl. nullable): {len(step2_mandatory)}",
            "",
            "=" * 72,
        ]
    )
    # Prefer pipeline-scoped inherited list for the classification block.
    class_scope = list(
        dict.fromkeys(
            list(nullable_res.present_explicit(nullable_scope))
            + [c for c in nullable_res.present_inherited(nullable_scope)
               if c in set(pipe_cols) or c in frame.columns]
            + list(pipe_cols)
            + [str(c) for c in frame.columns]
        )
    )
    for ln in format_nullable_classification(
        nullable_res,
        column_names=class_scope,
        max_inherited=60,
    ):
        lines.append(ln)
    lines.append(
        "(Excluded from Step-2 No-Null; not Class-2 pipeline bugs.)"
    )
    if lineage_gap_warnings:
        lines.extend(
            [
                "",
                "-" * 72,
                "Lineage Validation Warnings",
                "-" * 72,
            ]
        )
        for w in lineage_gap_warnings[:30]:
            lines.append(f"  ⚠ {w}")
        if len(lineage_gap_warnings) > 30:
            lines.append(f"  … and {len(lineage_gap_warnings) - 30} more")

    lines.extend(
        [
            "",
            "=" * 72,
            "Pipeline NULL Summary",
            "=" * 72,
            "(unique incomplete rows among Step-2 mandatory columns)",
            "",
            f"  {_CLASS_LABEL[CLASS_INHERITED]:<32} {summary_rows[CLASS_INHERITED]:>10,}",
            f"  {_CLASS_LABEL[CLASS_PIPELINE]:<32} {summary_rows[CLASS_PIPELINE]:>10,}",
            f"  {_CLASS_LABEL[CLASS_MATH]:<32} {summary_rows[CLASS_MATH]:>10,}",
            f"  {_CLASS_LABEL[CLASS_WARMUP]:<32} {summary_rows[CLASS_WARMUP]:>10,}",
            f"  {_CLASS_LABEL[CLASS_UNEXPECTED]:<32} {summary_rows[CLASS_UNEXPECTED]:>10,}",
            f"  {'TOTAL incomplete':<32} {would_remove if not filter_applied else removed_filter:>10,}",
            "",
            "Class 1 Inherited   -  parent NULL or Inherited Nullable chain.",
            "Class 2 Pipeline    -  NEW NULLs on non-nullable lineage; debug these.",
            "Class 3 Math        -  divide-by-zero etc.; not a code/registry bug.",
            "Warm-up             -  expected until windows fill (non-nullable parents).",
        ]
    )

    # --- Top NEW pipeline NULLs (actionable) ---
    # Only features that are neither Explicit nor Inherited Nullable.
    new_ranked = [
        r
        for r in feature_rows
        if int(r["new_null"]) > 0
        and r["null_class"] != CLASS_WARMUP
        and not r.get("inherited_nullable")
        and r["null_class"] != CLASS_INHERITED
    ]
    new_ranked.sort(
        key=lambda r: (
            0 if r["null_class"] == CLASS_PIPELINE else 1,
            -int(r["new_null"]),
            -int(r["exclusive"]),
            str(r["feature"]),
        )
    )

    lines.extend(
        [
            "",
            "-" * 72,
            "Top NEW Pipeline NULLs (debug these first)",
            "-" * 72,
        ]
    )
    if not new_ranked:
        lines.append("  (none  -  no actionable pipeline-created NULLs)")
    else:
        lines.append(
            f"  {'Feature':<42} {'New NULL':>10}  Primary class"
        )
        for r in new_ranked[: max(1, int(top_n))]:
            new_n = int(r["new_null"])
            lines.append(
                f"  {r['feature']:<42} {new_n:>10,}  "
                f"{_CLASS_LABEL.get(r['null_class'], r['null_class'])}"
            )
            lines.append(f"      {r['class'].get('reason')}")
            if r["class"].get("action"):
                lines.append(f"      -> {r['class'].get('action')}")

    # --- Mathematical ---
    math_ranked = [
        r
        for r in feature_rows
        if (r["null_class"] == CLASS_MATH or int(r["math"]) > 0)
        and not r.get("inherited_nullable")
    ]
    math_ranked.sort(key=lambda r: (-int(r["math"] or 0), str(r["feature"])))
    if math_ranked:
        lines.extend(
            [
                "",
                "-" * 72,
                "Mathematical NULLs (divide-by-zero etc.)",
                "-" * 72,
            ]
        )
        for r in math_ranked[:12]:
            lines.append(
                f"  {r['feature']:<42} div0={int(r['math']):>8,}  "
                f"null={r['null_rows']:,}"
            )

    # --- Inherited (brief; not the focus) ---
    inh_ranked = [
        r
        for r in feature_rows
        if r["null_class"] == CLASS_INHERITED or r.get("inherited_nullable")
    ]
    inh_ranked.sort(
        key=lambda r: (-int(r["inherited"] or r["null_rows"]), str(r["feature"]))
    )
    lines.extend(
        [
            "",
            "-" * 72,
            "Inherited / Inherited Nullable (not a pipeline problem  -  sample)",
            "-" * 72,
        ]
    )
    if not inh_ranked:
        lines.append("  (none)")
    else:
        for r in inh_ranked[:12]:
            tag = (
                "Inherited Nullable"
                if r.get("inherited_nullable")
                else "parent NULL"
            )
            lines.append(
                f"  {r['feature']:<42} "
                f"{tag}  null={int(r['null_rows']):,}"
            )

    # --- Warm-up sample (non-inherited-nullable only) ---
    warm_ranked = [
        r
        for r in feature_rows
        if r["null_class"] == CLASS_WARMUP and not r.get("inherited_nullable")
    ]
    warm_ranked.sort(key=lambda r: (-int(r["null_rows"]), str(r["feature"])))
    if warm_ranked:
        lines.extend(
            [
                "",
                "-" * 72,
                "Warm-up NULLs (expected  -  sample)",
                "-" * 72,
            ]
        )
        for r in warm_ranked[:8]:
            lines.append(f"  {r['feature']:<42} {r['null_rows']:>10,}")

    # Dependency chains for top NEW features
    chain_n = max(0, int(chain_examples))
    chain_rows = [r for r in new_ranked if r.get("lineage")][:chain_n]
    if not chain_rows:
        chain_rows = [r for r in feature_rows if r.get("lineage") and r["null_class"] == CLASS_PIPELINE][:chain_n]
    if chain_rows:
        lines.extend(
            [
                "",
                "-" * 72,
                "Dependency chains (NEW / actionable features)",
                "-" * 72,
            ]
        )
        for r in chain_rows:
            feat = r["feature"]
            lin = r["lineage"] or {}
            parents = list(lin.get("parents") or [])
            lines.append("")
            lines.append(feat)
            if parents:
                lines.append("depends on")
                for p in parents:
                    lines.append(f"    {p}")
            lines.append("↓")
            for p in parents:
                if p not in frame.columns:
                    lines.append(f"{p}  (not in frame)")
                    continue
                pn = int(frame[p].isna().sum())
                tag = (
                    f"pipeline:{lineage[p].get('transform')}"
                    if p in lineage
                    else "Registry/parent"
                )
                lines.append(f"{p} NULL")
                lines.append(f"{pn:,}  [{tag}]")
                lines.append("↓")
            lines.append(f"{feat} NULL")
            lines.append(f"{r['null_rows']:,}")
            cls = r["class"]
            if cls.get("propagated_any_parent"):
                lines.append(
                    f"  inherited from parent NULL: {cls['propagated_any_parent']:,}"
                )
            if cls.get("new_null_parents_present"):
                lines.append(
                    f"  NEW (parents present): {cls['new_null_parents_present']:,}"
                )
            if cls.get("divide_by_zero"):
                lines.append(f"  mathematical div0: {cls['divide_by_zero']:,}")
            lines.append(f"  class: {_CLASS_LABEL.get(r['null_class'], r['null_class'])}")

    # Verdict
    lines.extend(["", "-" * 72, "Verdict", "-" * 72])
    n_new = summary_rows[CLASS_PIPELINE]
    n_inh = summary_rows[CLASS_INHERITED]
    n_math = summary_rows[CLASS_MATH]
    n_warm = summary_rows[CLASS_WARMUP]
    if would_remove <= 0 and not filter_applied:
        lines.append("No incomplete pipeline rows  -  nothing for Step-2 No-Null to remove.")
    elif n_new > max(n_inh, 1) * 0.25:
        lines.append(
            f"Focus on Class 2: {n_new:,} incomplete rows driven by NEW pipeline NULLs. "
            "Use 'Top NEW Pipeline NULLs' above."
        )
    elif n_inh >= n_new and n_inh > 0:
        lines.append(
            f"Most incomplete rows are Class 1 Inherited ({n_inh:,}). "
            "Pipeline interactions are mostly innocent  -  Registry NULL policy "
            f"and warm-up ({n_warm:,}) explain the bulk."
        )
        if n_new:
            lines.append(
                f"Still investigate {n_new:,} Class 2 pipeline-created rows."
            )
    else:
        lines.append(
            f"Inherited={n_inh:,}, Pipeline-created={n_new:,}, "
            f"Math={n_math:,}, Warm-up={n_warm:,}."
        )
    if n_math:
        lines.append(
            f"Mathematical NULLs ({n_math:,} rows) are separate  -  not Registry or code bugs."
        )
    lines.append("=" * 72)
    return "\n".join(lines)


def emit_pipeline_no_null_report_lines(
    report_text: str,
    *,
    progress_fn: Any | None = None,
) -> list[str]:
    """Split report into log lines; optionally stream via progress_fn(msg, 0, 0)."""
    lines = [ln for ln in str(report_text or "").splitlines()]
    if progress_fn is not None:
        for ln in lines:
            try:
                progress_fn(ln, 0, 0)
            except TypeError:
                try:
                    progress_fn(ln)
                except Exception:
                    pass
            except Exception:
                pass
    return lines
