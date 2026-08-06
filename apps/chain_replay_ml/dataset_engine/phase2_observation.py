"""Phase 2 observation — usage-based Dataset Engine evidence.

Runs Create Model under ``ARUNEO_DATASET_ENGINE=auto`` across several
configs and appends evidence to a JSONL log. Exit criteria are usage-based
(see dataset_engine README), not calendar days.

Usage (from angelone/chart)::

  set PYTHONPATH=.
  python -m chain_replay_ml.dataset_engine.phase2_observation
  python -m chain_replay_ml.dataset_engine.phase2_observation --summary
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from chain_replay_ml.dataset_engine.run_create_model_engine_ab import (
    _chart_data_dir,
    _load_ui_config,
    build_observation_config,
)


_LOG = logging.getLogger("phase2_observation")


def _evidence_dir(data_dir: Path) -> Path:
    d = data_dir / "dataset_engine_observation"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _evidence_jsonl(data_dir: Path) -> Path:
    return _evidence_dir(data_dir) / "phase2_create_model.jsonl"


def _intersect_features(data_dir: Path, dataset: str, features: list[str]) -> list[str]:
    path = data_dir / "datasets" / f"{dataset}.parquet"
    if not path.is_file():
        return list(features)
    try:
        import pyarrow.parquet as pq

        names = set(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return list(features)
    return [f for f in features if f in names]


def _datasets_with_sidecar(data_dir: Path) -> list[str]:
    """Analysis datasets that have both parquet + json (trainability gate)."""
    out: list[str] = []
    ds = data_dir / "datasets"
    if not ds.is_dir():
        return out
    for parquet in sorted(ds.glob("analysis_*.parquet")):
        name = parquet.stem
        if (ds / f"{name}.json").is_file():
            out.append(name)
    return out


def _scenarios(ui: dict[str, Any], data_dir: Path, stamp: str) -> list[dict[str, Any]]:
    """Diverse Create Model configs for Phase 2 / 3B coverage."""
    preferred = str(ui.get("dataset") or "analysis_206r_193p_3s_20260730_025644")
    available = _datasets_with_sidecar(data_dir)
    # Prefer mid then large known families when present.
    preferred_order = [
        "analysis_206r_193p_3s_20260730_025644",
        "analysis_206r_193p_3s_20260730_094409",
        preferred,
    ]
    ordered: list[str] = []
    for name in preferred_order + available:
        if name in available and name not in ordered:
            ordered.append(name)
    if not ordered:
        ordered = [preferred]

    mid = ordered[0]
    large = ordered[1] if len(ordered) > 1 else ordered[0]

    base_name = f"Phase2_{stamp}"
    ui_features = list(ui.get("features") or [])

    def _cfg(*, model_name: str, dataset: str, **overrides: Any) -> dict[str, Any]:
        cfg = build_observation_config(ui, data_dir=data_dir, model_name=model_name)
        cfg["dataset"] = dataset
        feats = _intersect_features(data_dir, dataset, ui_features)
        if feats:
            cfg["features"] = feats
        for k, v in overrides.items():
            if k == "premium":
                enabled, lo, hi = v
                cfg["premiumSelection"] = {
                    "enabled": bool(enabled),
                    "premium_min": float(lo),
                    "premium_max": float(hi),
                }
            elif k == "walk_forward":
                wf = dict(cfg.get("walkForward") or {})
                wf.update(v)
                cfg["walkForward"] = wf
            else:
                cfg[k] = v
        return cfg

    scenarios = [
        {
            "id": "ds_025644_binary_premium",
            "label": f"Dataset {mid} · binary + premium 15–100",
            "config": _cfg(
                model_name=f"{base_name}_025644_bin",
                dataset=mid,
                target="label_up_2pct_5m",
                predictionType="binary",
                prediction_type="binary",
                premium=(True, 15.0, 100.0),
                walk_forward={"optimization_metric": "f1"},
            ),
        },
    ]
    if large != mid:
        scenarios.append(
            {
                "id": "ds_094409_reg_premium_lean",
                "label": f"Dataset {large} · lean WF regression + premium 15–100",
                "config": _cfg(
                    model_name=f"{base_name}_094409_lean",
                    dataset=large,
                    target="future_ltp_5m",
                    predictionType="regression",
                    prediction_type="regression",
                    premium=(True, 15.0, 100.0),
                    walk_forward={
                        "n_folds": 3,
                        "train_window_size": 3000,
                        "validation_window_size": 800,
                        "feature_selection_method": "none",
                    },
                ),
            }
        )
        scenarios.append(
            {
                "id": "ds_094409_binary_premium_lean",
                "label": f"Dataset {large} · lean WF binary + premium 15–100",
                "config": _cfg(
                    model_name=f"{base_name}_094409_bin",
                    dataset=large,
                    target="label_up_2pct_5m",
                    predictionType="binary",
                    prediction_type="binary",
                    premium=(True, 15.0, 100.0),
                    walk_forward={
                        "n_folds": 3,
                        "train_window_size": 3000,
                        "validation_window_size": 800,
                        "feature_selection_method": "none",
                        "optimization_metric": "f1",
                    },
                ),
            }
        )
    return scenarios


def _extract_load(package_dir: str | None) -> dict[str, Any]:
    if not package_dir:
        return {}
    meta_path = Path(package_dir) / "metadata.json"
    if not meta_path.is_file():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    load = meta.get("dataset_load") or {}
    return load if isinstance(load, dict) else {}


def _run_scenario(
    *,
    data_dir: Path,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    # Phase 2 production posture: auto (Engine + fallback).
    os.environ["ARUNEO_DATASET_ENGINE"] = "auto"
    from chain_replay_ml.training.orchestrator import train_model

    cfg = scenario["config"]
    t0 = time.perf_counter()

    def on_progress(ev: dict[str, Any]) -> None:
        stage = ev.get("stage") or ""
        status = ev.get("status") or ""
        if status in ("done", "fail") or stage in ("preparing_dataset", "walk_forward", "saving"):
            _LOG.info(
                "progress scenario=%s stage=%s status=%s",
                scenario["id"],
                stage,
                status,
            )

    result = train_model(data_dir=str(data_dir), raw_config=cfg, on_progress=on_progress)
    result = dict(result or {})
    package_dir = result.get("package_dir")
    load = _extract_load(package_dir)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scenario_id": scenario["id"],
        "label": scenario["label"],
        "ok": bool(result.get("ok")),
        "wall_sec": round(time.perf_counter() - t0, 3),
        "model_name": cfg.get("modelName") or cfg.get("model_name"),
        "dataset": cfg.get("dataset"),
        "target": cfg.get("target"),
        "prediction_type": cfg.get("predictionType") or cfg.get("prediction_type"),
        "package_dir": package_dir,
        "error": result.get("error"),
        "blocked": result.get("blocked"),
        "validation": result.get("validation"),
        "env": "auto",
        "dataset_load": load,
        "engine_used": load.get("backend") == "dataset_engine" and not load.get("engine_fallback"),
        "engine_fallback": bool(load.get("engine_fallback")),
        "engine_fallback_reason": load.get("engine_fallback_reason"),
    }
    return entry


def _append_evidence(data_dir: Path, entry: dict[str, Any]) -> Path:
    path = _evidence_jsonl(data_dir)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    return path


def _seed_phase1_pair(data_dir: Path) -> int:
    """Import prior off/on Create Model pair into the Phase 2 log once."""
    report = data_dir / "models" / "create_model_engine_ab_20260731_015113.json"
    if not report.is_file():
        return 0
    path = _evidence_jsonl(data_dir)
    if path.is_file() and "phase1_create_model_pair" in path.read_text(encoding="utf-8"):
        return 0
    doc = json.loads(report.read_text(encoding="utf-8"))
    seeded = 0
    for flag, run in (doc.get("runs") or {}).items():
        pkg = run.get("package_dir")
        load = _extract_load(pkg) or run.get("dataset_load") or {}
        entry = {
            "ts": "2026-07-31T01:54:20",
            "scenario_id": f"phase1_create_model_pair_{flag}",
            "label": f"Phase 1 Create Model pair ({flag})",
            "ok": bool(run.get("ok")),
            "wall_sec": run.get("wall_sec"),
            "model_name": run.get("model_name"),
            "dataset": doc.get("dataset"),
            "target": "future_ltp_5m",
            "prediction_type": "regression",
            "package_dir": pkg,
            "error": run.get("error"),
            "env": flag,
            "dataset_load": load,
            "engine_used": load.get("backend") == "dataset_engine",
            "engine_fallback": bool(load.get("engine_fallback")),
            "engine_fallback_reason": load.get("engine_fallback_reason"),
            "source": "seed_phase1_ab",
            "transparent_replacement": bool((doc.get("compare") or {}).get("ok")),
        }
        _append_evidence(data_dir, entry)
        seeded += 1
    return seeded


def summarize(data_dir: Path) -> dict[str, Any]:
    path = _evidence_jsonl(data_dir)
    rows: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    ok_runs = [r for r in rows if r.get("ok")]
    auto_runs = [r for r in ok_runs if r.get("env") == "auto"]
    unexpected_fallback = [
        r
        for r in auto_runs
        if r.get("engine_fallback")
        and "duckdb" not in str(r.get("engine_fallback_reason") or "").lower()
        and "import" not in str(r.get("engine_fallback_reason") or "").lower()
    ]
    scenarios = sorted({r.get("scenario_id") for r in ok_runs if r.get("scenario_id")})
    datasets = sorted({r.get("dataset") for r in ok_runs if r.get("dataset")})
    targets = sorted({r.get("target") for r in ok_runs if r.get("target")})

    summary = {
        "evidence_path": str(path),
        "total_entries": len(rows),
        "successful_runs": len(ok_runs),
        "auto_successful_runs": len(auto_runs),
        "distinct_scenarios": scenarios,
        "distinct_datasets": datasets,
        "distinct_targets": targets,
        "unexpected_engine_fallbacks": len(unexpected_fallback),
        "unexpected_fallback_ids": [r.get("scenario_id") for r in unexpected_fallback],
        "engine_used_count": sum(1 for r in auto_runs if r.get("engine_used")),
        "criteria": {
            # Broader bar: ≥2 datasets before Phase 3B (MI/HCA/Discovery).
            "multi_config_coverage": len(scenarios) >= 3 and len(targets) >= 2,
            "multi_dataset_coverage": len(datasets) >= 2,
            "no_unexpected_fallbacks": len(unexpected_fallback) == 0 and len(auto_runs) > 0,
            "auto_runs_present": len(auto_runs) >= 2,
        },
    }
    summary["phase2_exit_ready"] = all(summary["criteria"].values())
    summary["phase3b_ready"] = bool(summary["phase2_exit_ready"])
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Phase 2 Dataset Engine observation runs")
    parser.add_argument("--data-dir", default=str(_chart_data_dir()))
    parser.add_argument("--ui-config", default="")
    parser.add_argument("--summary", action="store_true", help="Print evidence summary only")
    parser.add_argument(
        "--scenarios",
        default="all",
        help="Comma-separated scenario ids, or 'all'",
    )
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)

    seeded = _seed_phase1_pair(data_dir)
    if seeded:
        _LOG.info("Seeded %s Phase 1 evidence rows", seeded)

    if args.summary:
        summary = summarize(data_dir)
        print(json.dumps(summary, indent=2))
        return 0 if summary.get("phase2_exit_ready") else 2

    ui_path = Path(args.ui_config) if args.ui_config else data_dir / "ml_model_builder_config_tk.json"
    ui = _load_ui_config(ui_path)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    scenarios = _scenarios(ui, data_dir, stamp)
    wanted = {s.strip() for s in args.scenarios.split(",") if s.strip()}
    if "all" not in wanted:
        scenarios = [s for s in scenarios if s["id"] in wanted]

    failures = 0
    for sc in scenarios:
        _LOG.info("=== scenario %s: %s ===", sc["id"], sc["label"])
        entry = _run_scenario(data_dir=data_dir, scenario=sc)
        path = _append_evidence(data_dir, entry)
        _LOG.info(
            "recorded ok=%s engine_used=%s fallback=%s load=%s → %s",
            entry["ok"],
            entry["engine_used"],
            entry["engine_fallback"],
            {
                k: entry["dataset_load"].get(k)
                for k in (
                    "backend",
                    "load_time_sec",
                    "peak_rss_mb",
                    "rows_returned",
                    "columns_returned",
                )
            },
            path,
        )
        if not entry["ok"]:
            failures += 1

    summary = summarize(data_dir)
    summary_path = _evidence_dir(data_dir) / f"phase2_summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    _LOG.info("Wrote %s", summary_path)
    if failures:
        return 1
    return 0 if summary.get("phase2_exit_ready") else 0


if __name__ == "__main__":
    raise SystemExit(main())
