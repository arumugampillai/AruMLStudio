"""Run Create Model twice: Dataset Engine off then on (observation pair).

Uses the saved Create Model UI config as the base, with deterministic overrides
required for a fair transparent-replacement compare:

  - HPO disabled (nested walk-forward HPO off)
  - feature_selection_method = none
  - fixed model names (..._ENGINE_OFF / ..._ENGINE_ON)
  - smaller analysis dataset when present (025644) for practical runtime
  - regression future_ltp_5m + Premium Selection 15–100 (matches load A/B)

Usage (from angelone/chart)::

  set PYTHONPATH=.
  python -m chain_replay_ml.dataset_engine.run_create_model_engine_ab
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any


_LOG = logging.getLogger("create_model_engine_ab")


def _chart_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def _load_ui_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _pick_dataset(data_dir: Path, preferred: str) -> str:
    datasets = data_dir / "datasets"
    # Prefer the smaller known-good analysis set when available.
    candidates = [
        "analysis_206r_193p_3s_20260730_025644",
        preferred,
    ]
    for name in candidates:
        if (datasets / f"{name}.parquet").is_file():
            return name
    return preferred


def build_observation_config(
    ui: dict[str, Any],
    *,
    data_dir: Path,
    model_name: str,
) -> dict[str, Any]:
    """Clone UI Create Model JSON into a deterministic observation config."""
    cfg = json.loads(json.dumps(ui))  # deep copy via JSON
    preferred = str(cfg.get("dataset") or "")
    cfg["dataset"] = _pick_dataset(data_dir, preferred)
    cfg["modelName"] = model_name
    cfg["model_name"] = model_name
    # Regression holdout/WF metrics are the observation gate defaults.
    cfg["target"] = "future_ltp_5m"
    cfg["predictionType"] = "regression"
    cfg["prediction_type"] = "regression"
    # Determinism: no HPO / no RFE (identical matrices → identical metrics).
    cfg["hyperparameterOptimization"] = {
        **dict(cfg.get("hyperparameterOptimization") or {}),
        "enabled": False,
    }
    wf = dict(cfg.get("walkForward") or {})
    wf["feature_selection_method"] = "none"
    wf["optimization_metric"] = "composite"
    hpo = dict(wf.get("hyperparameter_optimization") or {})
    hpo["enabled"] = False
    wf["hyperparameter_optimization"] = hpo
    cfg["walkForward"] = wf
    cfg["validationStrategy"] = "walk_forward"
    # Exercise Engine premium pushdown (matches prior load A/B).
    prem = dict(cfg.get("premiumSelection") or {})
    prem["enabled"] = True
    prem["premium_min"] = float(prem.get("premium_min") or 15.0)
    prem["premium_max"] = float(prem.get("premium_max") or 100.0)
    cfg["premiumSelection"] = prem
    cfg["skipAuditValidation"] = True
    cfg["skip_dataset_audit"] = True
    cfg["skipDatasetValidation"] = True
    cfg["skip_dataset_validation"] = True
    return cfg


def _run_one(*, data_dir: Path, raw_config: dict[str, Any], engine_flag: str) -> dict[str, Any]:
    os.environ["ARUNEO_DATASET_ENGINE"] = engine_flag
    from chain_replay_ml.training.orchestrator import train_model

    t0 = time.perf_counter()

    def on_progress(ev: dict[str, Any]) -> None:
        stage = ev.get("stage") or ev.get("step") or ""
        status = ev.get("status") or ""
        if status in ("running", "done", "fail") or stage:
            _LOG.info("progress engine=%s stage=%s status=%s", engine_flag, stage, status)

    result = train_model(data_dir=str(data_dir), raw_config=raw_config, on_progress=on_progress)
    result = dict(result or {})
    result["engine_flag"] = engine_flag
    result["wall_sec"] = round(time.perf_counter() - t0, 3)
    result["model_name"] = raw_config.get("modelName") or raw_config.get("model_name")
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Create Model Dataset Engine off/on pair")
    parser.add_argument(
        "--data-dir",
        default=str(_chart_data_dir()),
        help="chart data directory (default: angelone/chart/data)",
    )
    parser.add_argument(
        "--ui-config",
        default="",
        help="Create Model UI JSON (default: data/ml_model_builder_config_tk.json)",
    )
    parser.add_argument(
        "--only",
        choices=("off", "on", "both"),
        default="both",
        help="Run one side or both (default both)",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    ui_path = Path(args.ui_config) if args.ui_config else data_dir / "ml_model_builder_config_tk.json"
    if not ui_path.is_file():
        _LOG.error("UI config not found: %s", ui_path)
        return 2

    ui = _load_ui_config(ui_path)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"DatasetEngine_CM_{stamp}"
    results: dict[str, Any] = {"stamp": stamp, "dataset": None, "runs": {}}

    order = []
    if args.only in ("off", "both"):
        order.append("off")
    if args.only in ("on", "both"):
        order.append("on")

    for flag in order:
        name = f"{base}_ENGINE_{flag.upper()}"
        cfg = build_observation_config(ui, data_dir=data_dir, model_name=name)
        results["dataset"] = cfg["dataset"]
        _LOG.info(
            "Starting Create Model engine=%s model=%s dataset=%s features=%s",
            flag,
            name,
            cfg["dataset"],
            len(cfg.get("features") or []),
        )
        run = _run_one(data_dir=data_dir, raw_config=cfg, engine_flag=flag)
        results["runs"][flag] = {
            "ok": bool(run.get("ok")),
            "model_name": run.get("model_name"),
            "package_dir": run.get("package_dir") or run.get("model_dir"),
            "wall_sec": run.get("wall_sec"),
            "error": run.get("error"),
            "blocked": run.get("blocked"),
        }
        _LOG.info(
            "Finished engine=%s ok=%s wall_sec=%s package=%s error=%s",
            flag,
            run.get("ok"),
            run.get("wall_sec"),
            results["runs"][flag]["package_dir"],
            run.get("error"),
        )
        if not run.get("ok"):
            print(json.dumps(results, indent=2, default=str))
            return 1

    off_pkg = results["runs"].get("off", {}).get("package_dir")
    on_pkg = results["runs"].get("on", {}).get("package_dir")
    if off_pkg and on_pkg:
        from chain_replay_ml.dataset_engine.compare_create_model_runs import (
            compare_create_model_packages,
            _print_report,
        )

        report = compare_create_model_packages(off_pkg, on_pkg)
        results["compare"] = report
        _print_report(report)
        out = data_dir / "datasets" / f"create_model_engine_ab_{stamp}.json"
        # keep report next to models is fine too
        out = data_dir / "models" / f"create_model_engine_ab_{stamp}.json"
        out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        _LOG.info("Wrote %s", out)
        return 0 if report.get("ok") else 1

    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
