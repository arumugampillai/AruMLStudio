"""Write Create Dataset / master-build timing + Numba verification reports."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT_DIR = os.path.join(_PKG_DIR, "benchmarks")


def default_report_paths(out_dir: str | None = None) -> tuple[str, str]:
    base = out_dir or _DEFAULT_OUT_DIR
    os.makedirs(base, exist_ok=True)
    return (
        os.path.join(base, "create_dataset_timing_report.json"),
        os.path.join(base, "create_dataset_timing_report.md"),
    )


def build_timing_report(
    *,
    numba_stats: dict[str, Any],
    phase_timings: dict[str, float],
    pipeline_stages: list[dict[str, Any]] | None = None,
    build_profiler_report: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a structured timing + Numba verification document."""
    total = float(phase_timings.get("create_dataset_wall_sec") or 0.0)
    phases = {k: round(float(v), 6) for k, v in phase_timings.items()}
    shares = {
        k: (round(100.0 * float(v) / total, 2) if total > 0 else None)
        for k, v in phases.items()
        if k != "create_dataset_wall_sec"
    }
    feature_sec = float(
        phases.get("feature_computation_sec")
        or phases.get("feature_generation_sec")
        or 0.0
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Create Dataset / master_build (raw ticks → features → master SQLite)",
        "notes": {
            "create_dataset_vs_prediction": (
                "Create Dataset (master_build) builds feature rows + prediction target "
                "columns (horizons) in one day path. Exporting a train-ready parquet from "
                "an existing master DB is a separate 'Create Dataset from Master' flow."
            ),
            "polars_duckdb": (
                "Per-row feature path is dict + controllers (not Polars). DuckDB/Polars "
                "appear mainly in dataset export / frame IO, not in build_feature_raw_for_row."
            ),
            "synthetic_benchmark": (
                "chain_replay_ml.performance.benchmark exercises controllers directly; "
                "this report is for the production Create Dataset entry points."
            ),
        },
        "numba": numba_stats,
        "phase_timings_sec": phases,
        "phase_share_pct": shares,
        "feature_engine_bottleneck": bool(
            total > 0 and feature_sec >= 0.4 * total
        ),
        "pipeline_stages": pipeline_stages or [],
        "build_profiler_top": _top_profiler_blocks(build_profiler_report),
        "meta": meta or {},
    }


def _top_profiler_blocks(report: dict[str, Any] | None, *, n: int = 12) -> list[dict[str, Any]]:
    if not report:
        return []
    ranked = report.get("ranked") or []
    if ranked:
        return [
            {
                "key": e.get("name"),
                "total_sec": float(e.get("total_sec") or 0),
                "rows": e.get("rows"),
            }
            for e in ranked[:n]
            if isinstance(e, dict)
        ]
    blocks = report.get("blocks") or report.get("functions") or []
    if isinstance(blocks, dict):
        items = [
            {"key": k, "total_sec": float((v or {}).get("total_sec") or 0), "rows": (v or {}).get("rows")}
            for k, v in blocks.items()
        ]
    else:
        items = [
            {
                "key": e.get("name") or e.get("key"),
                "total_sec": float(e.get("total_sec") or 0),
                "rows": e.get("rows"),
            }
            for e in blocks
            if isinstance(e, dict)
        ]
    items.sort(key=lambda x: float(x.get("total_sec") or 0), reverse=True)
    return items[:n]


def write_timing_report(
    doc: dict[str, Any],
    *,
    out_dir: str | None = None,
) -> tuple[str, str]:
    json_path, md_path = default_report_paths(out_dir)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(format_timing_markdown(doc))
    return json_path, md_path


def format_timing_markdown(doc: dict[str, Any]) -> str:
    numba = doc.get("numba") or {}
    phases = doc.get("phase_timings_sec") or {}
    shares = doc.get("phase_share_pct") or {}
    lines = [
        "# Create Dataset Timing Report",
        "",
        f"Generated: `{doc.get('generated_at')}`",
        "",
        f"**Scope:** {doc.get('scope')}",
        "",
        "## Numba (production path)",
        "",
        f"- Numba enabled: **{numba.get('numba_enabled_label', 'NO')}**",
        f"- `ARUNEO_FEATURE_NUMBA` env: `{numba.get('aruneo_feature_numba_env')}`",
        f"- Numba available: `{numba.get('numba_available')}`",
        f"- Numba kernel calls: `{numba.get('kernel_hits', 0):,}`",
        f"- Python fallback calls: `{numba.get('python_fallback_hits', 0):,}`",
        f"- Python fallback active: `{numba.get('python_fallback')}`"
        + (f" — {numba.get('python_fallback_reason')}" if numba.get("python_fallback_reason") else ""),
        "",
        "## Phase timings",
        "",
        "| Phase | Seconds | Share |",
        "|-------|--------:|------:|",
    ]
    order = [
        "loading_ticks_sec",
        "feature_computation_sec",
        "prediction_targets_sec",
        "sqlite_insert_sec",
        "polars_duckdb_sec",
        "write_output_sec",
        "create_dataset_wall_sec",
    ]
    labels = {
        "loading_ticks_sec": "Loading ticks",
        "feature_computation_sec": "Feature computation",
        "prediction_targets_sec": "Prediction targets",
        "sqlite_insert_sec": "DuckDB/SQLite insert",
        "polars_duckdb_sec": "Polars / DuckDB frame IO",
        "write_output_sec": "Writing output",
        "create_dataset_wall_sec": "Create Dataset wall",
    }
    seen: set[str] = set()
    for key in order:
        if key not in phases:
            continue
        seen.add(key)
        share = shares.get(key)
        share_s = f"{share:.1f}%" if share is not None else "—"
        lines.append(f"| {labels.get(key, key)} | {phases[key]:.3f} | {share_s} |")
    for key, val in phases.items():
        if key in seen:
            continue
        share = shares.get(key)
        share_s = f"{share:.1f}%" if share is not None else "—"
        lines.append(f"| {key} | {val:.3f} | {share_s} |")
    lines.extend(
        [
            "",
            f"**Feature engine likely bottleneck:** `{doc.get('feature_engine_bottleneck')}`",
            "",
            "## Notes",
            "",
        ]
    )
    for note in (doc.get("notes") or {}).values():
        lines.append(f"- {note}")
    top = doc.get("build_profiler_top") or []
    if top:
        lines.extend(["", "## Top build_profiler blocks", "", "| Key | Seconds |", "|-----|--------:|"])
        for item in top:
            lines.append(
                f"| {item.get('key') or item.get('name')} | {float(item.get('total_sec') or 0):.3f} |"
            )
    meta = doc.get("meta") or {}
    if meta:
        lines.extend(["", "## Meta", "", "```json", json.dumps(meta, indent=2), "```"])
    lines.append("")
    return "\n".join(lines)
