"""Confidence Label Builder — offline replay → continuous outcomes → binaries.

Strategy Simulator UI is unchanged. This module calls the shared replay engine
(``simulate_forced_entry_outcomes``) to generate Confidence Outcome Labels.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from chain_replay_ml.dataset_builder.writer import ensure_parquet_engine
from chain_replay_ml.strategy_registry.hashes import strategy_config_hash
from chain_replay_ml.strategy_registry.schema import normalize_strategy_config
from chain_replay_ml.strategy_registry.service import get_strategy_version
from chain_replay_ml.strategy_simulator.engine import simulate_forced_entry_outcomes
from chain_replay_ml.strategy_simulator.lab_source import load_lab_prediction_rows_for_simulation

from .confidence_dataset import confidence_dataset_paths, confidence_package_dir
from .target_spec import (
    CONTINUOUS_OUTCOME_FIELDS,
    derive_binary_labels,
    replay_label_columns,
)

ProgressCb = Callable[[dict[str, Any]], None]

REPLAY_MODE_FORCED = "forced_entry"
REPLAY_MODE_LIVE = "live_rules"  # reserved — not implemented in v1

# String / categorical columns that must survive parquet round-trip.
# (dataset_builder._write_parquet coerces unknown cols to float64 — wrong here.)
_OUTCOME_TEXT_COLS = frozenset({
    "prediction_id",
    "trading_day",
    "token",
    "exit_reason",
    "option_type",
})


def _write_label_parquet(df: pd.DataFrame, path: str) -> None:
    """Persist Label Builder frames without numeric-only coercion."""
    ensure_parquet_engine()
    safe = df.copy()
    for col in _OUTCOME_TEXT_COLS:
        if col not in safe.columns:
            continue
        safe[col] = [
            None
            if v is None or (isinstance(v, float) and pd.isna(v))
            else str(v)
            for v in safe[col].tolist()
        ]
    safe.to_parquet(path, index=False)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def label_runs_dir(lab_db_path: str) -> str:
    return os.path.join(confidence_package_dir(lab_db_path), "label_runs")


def latest_label_run_meta_path(lab_db_path: str) -> str:
    return os.path.join(label_runs_dir(lab_db_path), "latest.json")


def label_run_dir(lab_db_path: str, label_run_id: str) -> str:
    return os.path.join(label_runs_dir(lab_db_path), label_run_id)


def read_latest_label_run(lab_db_path: str) -> dict[str, Any] | None:
    path = latest_label_run_meta_path(lab_db_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def load_replay_outcome_frames(
    lab_db_path: str,
    *,
    label_run_id: str | None = None,
) -> dict[str, Any]:
    """Load continuous outcomes + derived binary labels from a label run."""
    meta = None
    if label_run_id:
        run_dir = label_run_dir(lab_db_path, label_run_id)
        meta_path = os.path.join(run_dir, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
    else:
        meta = read_latest_label_run(lab_db_path)
        if meta:
            label_run_id = str(meta.get("label_run_id") or "")
            run_dir = label_run_dir(lab_db_path, label_run_id) if label_run_id else ""
        else:
            run_dir = ""

    if not meta or not label_run_id:
        return {"ok": False, "error": "No Confidence Label Run found.", "meta": None}

    run_dir = label_run_dir(lab_db_path, str(label_run_id))
    outcomes_pq = os.path.join(run_dir, "outcomes.parquet")
    labels_pq = os.path.join(run_dir, "binary_labels.parquet")
    if not os.path.isfile(outcomes_pq):
        return {"ok": False, "error": f"Outcomes missing: {outcomes_pq}", "meta": meta}

    outcomes = pd.read_parquet(outcomes_pq)
    binaries = pd.read_parquet(labels_pq) if os.path.isfile(labels_pq) else None
    return {
        "ok": True,
        "meta": meta,
        "label_run_id": label_run_id,
        "outcomes": outcomes,
        "binary_labels": binaries,
        "run_dir": run_dir,
    }


def list_confidence_label_runs(lab_db_path: str) -> list[dict[str, Any]]:
    """All Confidence Label Runs for a lab, newest first."""
    root = label_runs_dir(lab_db_path)
    if not os.path.isdir(root):
        return []
    latest = read_latest_label_run(lab_db_path) or {}
    latest_id = str(latest.get("label_run_id") or "")
    runs: list[dict[str, Any]] = []
    for name in os.listdir(root):
        run_path = os.path.join(root, name)
        meta_path = os.path.join(run_path, "meta.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        rid = str(meta.get("label_run_id") or name)
        meta = dict(meta)
        meta["label_run_id"] = rid
        meta["is_latest"] = rid == latest_id
        meta["run_dir"] = run_path
        runs.append(meta)
    runs.sort(key=lambda m: str(m.get("created_at") or ""), reverse=True)
    return runs


def assess_label_run_staleness(
    lab_db_path: str,
    *,
    data_dir: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compare a Label Run to the current Prediction Dataset + strategy config.

    Stale when prediction hash drifts or strategy_config_hash no longer matches.
    """
    from chain_replay_ml.model_lab.prediction_builder import prediction_dataset_status

    doc = meta if meta is not None else read_latest_label_run(lab_db_path)
    if not doc:
        return {
            "ok": True,
            "has_run": False,
            "up_to_date": False,
            "status": "missing",
            "status_display": "Not built",
            "reasons": ["No Confidence Labels yet — build Replay-Based labels first."],
            "rebuild_recommended": True,
        }

    reasons: list[str] = []
    status = prediction_dataset_status(lab_db_path, light=True)
    cur_hash = str(status.get("dataset_hash") or "")
    run_hash = str(doc.get("prediction_dataset_hash") or "")
    cur_rows = int(status.get("row_count") or 0)
    run_rows = int(doc.get("prediction_row_count") or doc.get("rows_loaded") or 0)

    if cur_hash and run_hash and cur_hash != run_hash:
        reasons.append("Prediction Dataset changed after this Label Run.")
    elif cur_rows and run_rows and cur_rows != run_rows:
        reasons.append(
            f"Prediction Dataset row count changed ({run_rows:,} → {cur_rows:,})."
        )

    version_id = str(doc.get("strategy_version_id") or "")
    run_cfg_hash = str(doc.get("strategy_config_hash") or "")
    strategy_id = str(doc.get("strategy_id") or "")
    version = get_strategy_version(data_dir, version_id) if version_id else None
    if not version:
        reasons.append("Strategy version for this Label Run is no longer available.")
        current_cfg_hash: str | None = None
    else:
        if not strategy_id:
            strategy_id = str(version.get("strategy_id") or "")
        # Prefer champion config: a new strategy version invalidates prior labels.
        current_cfg_hash = str(version.get("config_hash") or "") or None
        if strategy_id:
            try:
                from chain_replay_ml.strategy_registry import get_strategy_detail

                detail = get_strategy_detail(data_dir, strategy_id)
                champ = (detail or {}).get("champion_version") or {}
                champ_hash = str(champ.get("config_hash") or "")
                if champ_hash:
                    current_cfg_hash = champ_hash
            except Exception:
                pass
        if run_cfg_hash and current_cfg_hash and run_cfg_hash != current_cfg_hash:
            reasons.append("Strategy configuration changed.")

    up_to_date = not reasons
    return {
        "ok": True,
        "has_run": True,
        "up_to_date": up_to_date,
        "status": "up_to_date" if up_to_date else "out_of_date",
        "status_display": "Up to date" if up_to_date else "Out of date",
        "reasons": reasons,
        "rebuild_recommended": not up_to_date,
        "label_run_id": doc.get("label_run_id"),
        "strategy_version_id": version_id,
        "strategy_config_hash": run_cfg_hash,
        "prediction_dataset_hash": run_hash,
        "current_prediction_dataset_hash": cur_hash or None,
        "current_strategy_config_hash": current_cfg_hash,
    }


def outcome_summary_from_frame(outcomes: pd.DataFrame | None) -> dict[str, Any]:
    if outcomes is None or outcomes.empty:
        return {
            "rows_processed": 0,
            "avg_net_pnl": None,
            "avg_return_pct": None,
            "avg_holding_seconds": None,
        }

    def _avg(col: str) -> float | None:
        if col not in outcomes.columns:
            return None
        s = pd.to_numeric(outcomes[col], errors="coerce")
        return round(float(s.mean()), 4) if s.notna().any() else None

    hold = _avg("holding_seconds")
    return {
        "rows_processed": int(len(outcomes)),
        "avg_net_pnl": _avg("net_pnl"),
        "avg_return_pct": _avg("return_pct"),
        "avg_holding_seconds": round(float(hold), 3) if hold is not None else None,
    }


def confidence_labels_status(
    lab_db_path: str,
    *,
    data_dir: str,
) -> dict[str, Any]:
    """Payload for the Confidence Labels page."""
    from .target_spec import REPLAY_TARGET_SPECS

    runs = list_confidence_label_runs(lab_db_path)
    latest = read_latest_label_run(lab_db_path)
    stale = assess_label_run_staleness(lab_db_path, data_dir=data_dir, meta=latest)

    targets = [
        {
            "key": t.key,
            "label": t.label,
            "column": t.column,
            "rule": t.rule,
        }
        for t in REPLAY_TARGET_SPECS
    ]

    summary = (latest or {}).get("outcome_summary")
    if latest and not summary:
        loaded = load_replay_outcome_frames(
            lab_db_path, label_run_id=str(latest.get("label_run_id") or "") or None
        )
        if loaded.get("ok"):
            summary = outcome_summary_from_frame(loaded.get("outcomes"))

    return {
        "ok": True,
        "latest": latest,
        "runs": runs,
        "staleness": stale,
        "replay_targets": targets,
        "outcome_summary": summary,
        "run_count": len(runs),
    }


def run_confidence_label_builder(
    lab_db_path: str,
    *,
    data_dir: str,
    strategy_version_id: str,
    replay_mode: str = REPLAY_MODE_FORCED,
    on_progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """
    Replay Prediction Dataset once → continuous outcomes → Replay-Based binaries.

    Stores artifacts under ``{lab}_confidence/label_runs/<id>/`` and updates
    ``latest.json``. Reproducibility meta: strategy_version_id + strategy_config_hash.
    """
    def _prog(payload: dict[str, Any]) -> None:
        if on_progress:
            try:
                on_progress(payload)
            except Exception:
                pass

    if replay_mode != REPLAY_MODE_FORCED:
        return {
            "ok": False,
            "error": (
                f"Replay mode '{replay_mode}' is not implemented yet. "
                f"Use '{REPLAY_MODE_FORCED}'."
            ),
        }

    if not lab_db_path or not os.path.isfile(lab_db_path):
        return {"ok": False, "error": "Prediction Dataset / Research Lab DB not found."}

    version = get_strategy_version(data_dir, strategy_version_id)
    if not version:
        return {"ok": False, "error": f"Strategy version not found: {strategy_version_id}"}

    cfg = normalize_strategy_config(version.get("config") or {})
    cfg_hash = str(version.get("config_hash") or strategy_config_hash(cfg))

    _prog({"phase": "load", "message": "Loading prediction rows…"})
    rows = load_lab_prediction_rows_for_simulation(lab_db_path)
    if not rows:
        return {"ok": False, "error": "Prediction Dataset is empty."}

    from chain_replay_ml.model_lab.prediction_builder import prediction_dataset_status

    status = prediction_dataset_status(lab_db_path, light=True)
    pred_hash = str(status.get("dataset_hash") or "")
    pred_rows = int(status.get("row_count") or len(rows))

    _prog({
        "phase": "replay",
        "message": f"Forced-entry replay · {len(rows):,} rows…",
        "rows": len(rows),
    })
    outcomes, stats = simulate_forced_entry_outcomes(
        rows,
        cfg=cfg,
        strategy_version_id=strategy_version_id,
    )

    _prog({"phase": "derive", "message": "Deriving Replay-Based binary labels…"})
    binaries = derive_binary_labels(outcomes)

    label_run_id = uuid.uuid4().hex
    run_dir = label_run_dir(lab_db_path, label_run_id)
    os.makedirs(run_dir, exist_ok=True)

    ensure_parquet_engine()
    outcomes_df = pd.DataFrame(outcomes)
    # Keep stable column order when present
    ordered = [c for c in CONTINUOUS_OUTCOME_FIELDS if c in outcomes_df.columns]
    extra = [c for c in outcomes_df.columns if c not in ordered]
    outcomes_df = outcomes_df[ordered + extra]
    binaries_df = pd.DataFrame(binaries)

    outcomes_pq = os.path.join(run_dir, "outcomes.parquet")
    labels_pq = os.path.join(run_dir, "binary_labels.parquet")
    _write_label_parquet(outcomes_df, outcomes_pq)
    _write_label_parquet(binaries_df, labels_pq)

    # Positive rates for report
    pos_rates: dict[str, float | None] = {}
    for col in replay_label_columns():
        if col not in binaries_df.columns:
            continue
        s = pd.to_numeric(binaries_df[col], errors="coerce")
        scored = s.dropna()
        pos_rates[col] = (
            round(float((scored == 1).mean()), 4) if len(scored) else None
        )

    outcome_summary = outcome_summary_from_frame(outcomes_df)

    meta = {
        "label_run_id": label_run_id,
        "created_at": _utc_now(),
        "lab_db_path": os.path.abspath(lab_db_path),
        "strategy_version_id": strategy_version_id,
        "strategy_config_hash": cfg_hash,
        "strategy_id": version.get("strategy_id"),
        "strategy_display_name": version.get("display_name"),
        "strategy_version_label": version.get("version_label"),
        "replay_mode": replay_mode,
        "prediction_dataset_hash": pred_hash,
        "prediction_row_count": pred_rows,
        "rows_loaded": len(rows),
        "outcomes_written": int(stats.get("outcomes") or 0),
        "stats": stats,
        "binary_columns": list(replay_label_columns()),
        "positive_rates": pos_rates,
        "outcome_summary": outcome_summary,
        "outcomes_parquet": outcomes_pq,
        "binary_labels_parquet": labels_pq,
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    latest_path = latest_label_run_meta_path(lab_db_path)
    os.makedirs(os.path.dirname(latest_path), exist_ok=True)
    with open(latest_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    # Also mirror into confidence dataset paths for discoverability
    paths = confidence_dataset_paths(lab_db_path)
    os.makedirs(paths["datasets_dir"], exist_ok=True)
    sidecar = os.path.join(paths["datasets_dir"], "latest_label_run.json")
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    _prog({"phase": "done", "message": "Confidence Label Run complete.", **meta})
    return {"ok": True, **meta}
