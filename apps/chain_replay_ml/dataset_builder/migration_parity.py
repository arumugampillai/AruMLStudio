"""Parity harness: Master historical columns vs Transformation Pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .feature_migration import (
    get_migration_family,
    horizons_compatible_with_interval,
    mark_family_parity,
)
from .transformations import run_transformation_pipeline
from .transformations.base import TransformContext


def _master_style_ratio_lags(
    frame: pd.DataFrame,
    *,
    base: str,
    horizons: list[tuple[str, float]],
    sample_interval_sec: float,
    partition_by: list[str],
) -> pd.DataFrame:
    """Simulate Master lag columns as row-shifts of the sampled base ratio.

    Pipeline parity uses the same row-based semantics (no calendar lookback).
    On a uniform gap-free grid this matches values at sample timestamps T−Δ.
    """
    out = frame.copy()
    for suffix, sec in horizons:
        rows = int(round(float(sec) / float(sample_interval_sec)))
        col = f"{base}_lag_{suffix}"
        if partition_by:
            out[col] = out.groupby(partition_by, sort=False, group_keys=False)[base].transform(
                lambda s, n=rows: s.shift(n)
            )
        else:
            out[col] = out[base].shift(rows)
    return out


def compare_lag_family_parity(
    frame: pd.DataFrame,
    *,
    family_id: str = "ltp_to_spot_ratio",
    sample_interval_sec: float = 3.0,
    partition_by: list[str] | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    update_status: bool = True,
) -> dict[str, Any]:
    """Compare pipeline Lag output to Master-style lag columns.

    ``frame`` must contain the family base feature (and partition columns).
    Master reference columns are built as row-shifts of that base (uniform-grid
    equivalence). Pipeline uses Master-compatible suffixes via family config.
    """
    fam = get_migration_family(family_id)
    parts = partition_by or ["trading_day", "token"]
    base = fam.base_feature
    if base not in frame.columns:
        raise KeyError(f"Base feature {base!r} missing from frame")

    ok, bad = horizons_compatible_with_interval(fam.horizons, sample_interval_sec)
    detail: dict[str, Any] = {
        "family_id": family_id,
        "sample_interval_sec": sample_interval_sec,
        "compatible_horizons": [s for s, _ in ok],
        "blocked_horizons": [s for s, _ in bad],
        "columns": {},
    }

    if not ok:
        result = {
            "ok": False,
            "status": "blocked",
            "detail": detail,
            "message": (
                f"No horizons divisible by sample_interval_sec={sample_interval_sec}. "
                f"Blocked: {detail['blocked_horizons']}"
            ),
        }
        if update_status:
            mark_family_parity(
                family_id,
                status="blocked",
                detail=detail,
                notes=result["message"],
            )
        return result

    # Restrict family pipeline config to compatible horizons only for the run,
    # but report blocked members so the family is not falsely marked passed.
    cfg = fam.pipeline_config(
        sample_interval_sec=sample_interval_sec,
        partition_by=parts,
    )
    # Filter horizons in config to compatible only
    cfg["transformations"][0]["params"]["horizons"] = [
        {"seconds": float(sec), "suffix": str(suffix)} for suffix, sec in ok
    ]

    master_ref = _master_style_ratio_lags(
        frame,
        base=base,
        horizons=ok,
        sample_interval_sec=sample_interval_sec,
        partition_by=parts,
    )
    ctx = TransformContext(config=cfg, sample_interval_sec=sample_interval_sec)
    pipe = run_transformation_pipeline(frame[[*parts, base]].copy(), cfg, context=ctx)

    all_pass = True
    for suffix, _sec in ok:
        col = f"{base}_lag_{suffix}"
        if col not in pipe.frame.columns:
            detail["columns"][col] = {"pass": False, "error": "missing pipeline column"}
            all_pass = False
            continue
        a = master_ref[col]
        b = pipe.frame[col]
        # Compare where both non-null
        both = a.notna() & b.notna()
        if int(both.sum()) == 0:
            detail["columns"][col] = {"pass": False, "error": "no overlapping non-null rows"}
            all_pass = False
            continue
        close = np.isclose(
            a[both].to_numpy(dtype=float),
            b[both].to_numpy(dtype=float),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        null_match = bool((a.isna() == b.isna()).all())
        passed = bool(close.all()) and null_match
        detail["columns"][col] = {
            "pass": passed,
            "compared_rows": int(both.sum()),
            "max_abs_err": float(np.max(np.abs(a[both] - b[both]))) if len(a[both]) else None,
            "null_mask_match": null_match,
        }
        if not passed:
            all_pass = False

    if bad:
        status = "partial" if all_pass else "failed"
        message = (
            f"Compatible horizons {'passed' if all_pass else 'failed'}; "
            f"blocked (not divisible by {sample_interval_sec}s): "
            f"{detail['blocked_horizons']}"
        )
        # Family not fully pipeline-owned until every horizon passes.
        ok_final = False
    else:
        status = "passed" if all_pass else "failed"
        message = "All horizons matched" if all_pass else "Parity mismatches"
        ok_final = all_pass

    result = {
        "ok": ok_final,
        "status": status,
        "detail": detail,
        "message": message,
    }
    if update_status:
        mark_family_parity(
            family_id,
            status=status,  # type: ignore[arg-type]
            detail=detail,
            notes=message,
        )
        # Only mark pipeline_owned when fully passed (mark_family_parity does that).
        if status != "passed":
            fam = get_migration_family(family_id)
            fam.pipeline_owned = False
    return result


__all__ = [
    "compare_lag_family_parity",
]
