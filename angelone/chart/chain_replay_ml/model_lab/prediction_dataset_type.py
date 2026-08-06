"""Per-model Seen / Unseen classification for Prediction Dataset trading days.

Seen  = trading days used in training and/or walk-forward validation for this model.
Unseen = trading days present for the lab (parent / Master) that were never used
         by this model.

Classification is per Research Lab (per model), never global.
"""

from __future__ import annotations

import os
import re
from typing import Any, Mapping

from .prediction_schema import DATASET_TYPE_SEEN, DATASET_TYPE_UNSEEN

_DAY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _add_day(out: set[str], value: Any) -> None:
    text = str(value or "").strip()
    if not text or text == "—":
        return
    if _DAY_RE.fullmatch(text):
        out.add(text)
        return
    for match in _DAY_RE.findall(text):
        out.add(match)


def _add_days_from_blocks(out: set[str], blocks: Any) -> None:
    if isinstance(blocks, list):
        for item in blocks:
            if isinstance(item, dict):
                _add_day(out, item.get("trading_day") or item.get("day"))
            else:
                _add_day(out, item)
    elif isinstance(blocks, dict):
        for key, val in blocks.items():
            if _DAY_RE.fullmatch(str(key or "").strip()):
                _add_day(out, key)
            _add_day(out, val)


def _collect_from_mapping(out: set[str], doc: Mapping[str, Any] | None) -> None:
    if not isinstance(doc, Mapping):
        return
    # Prefer concrete day lists that reflect what was actually exported / trained.
    # Do not harvest trading_day_filter.selected_dates — those are pre-filter checkboxes
    # and can include expiry (or other) days that never entered the training parquet.
    for key in (
        "trading_day_labels",
        "trading_days_label",
        "trading_day",
        "training_days",
        "train_days",
        "validation_days",
        "val_days",
        "exported_dates",
        "exported_days",
        "included_dates",
    ):
        if key in doc:
            raw = doc.get(key)
            if isinstance(raw, (list, tuple, set)):
                for item in raw:
                    _add_day(out, item)
            else:
                _add_day(out, raw)
    if "trading_days" in doc:
        raw = doc.get("trading_days")
        if isinstance(raw, (list, tuple, set)):
            for item in raw:
                _add_day(out, item)
        # integer trading_days counts are ignored
    _add_days_from_blocks(out, doc.get("days"))
    _add_days_from_blocks(out, doc.get("sources"))
    mf = doc.get("master_filter")
    if isinstance(mf, Mapping):
        # Only trust selected_days when the export was an explicit subset (not all_days).
        if not bool(mf.get("all_days")):
            sel = mf.get("selected_days")
            if isinstance(sel, (list, tuple, set)):
                for item in sel:
                    _add_day(out, item)
            _add_day(out, mf.get("trading_day"))
    tdf = doc.get("trading_day_filter")
    if isinstance(tdf, Mapping):
        # Exported / included dates only — never selected_dates (pre-filter).
        for key in ("exported_dates", "included_dates", "training_dates"):
            raw = tdf.get(key)
            if isinstance(raw, (list, tuple, set)):
                for item in raw:
                    _add_day(out, item)
            elif isinstance(raw, str):
                _add_day(out, raw)


def resolve_model_master_filter(
    lab: Any = None,
    *,
    parent_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Training/export Master filter criteria for this model (premium, ATM, …).

    Preference order:
    1) lab.dataset_snapshot.dataset_build_snapshot.master_filter
    2) lab.dataset_snapshot.master_filter / selection_method.criteria
    3) parent dataset registry ``master_filter`` / ``selection_method.criteria``
    """
    candidates: list[Mapping[str, Any]] = []

    def _push(doc: Any) -> None:
        if not isinstance(doc, Mapping):
            return
        mf = doc.get("master_filter")
        if isinstance(mf, Mapping) and mf:
            candidates.append(mf)
        sm = doc.get("selection_method")
        if isinstance(sm, Mapping):
            crit = sm.get("criteria")
            if isinstance(crit, Mapping) and crit:
                candidates.append(crit)

    snap = getattr(lab, "dataset_snapshot", None) if lab is not None else None
    if isinstance(snap, Mapping):
        nested = snap.get("dataset_build_snapshot")
        if isinstance(nested, Mapping):
            _push(nested)
        _push(snap)
    if isinstance(parent_meta, Mapping):
        _push(parent_meta)

    if not candidates:
        return {}
    # First non-empty wins (lab snapshot is preferred via candidates order).
    raw = dict(candidates[0])
    out: dict[str, Any] = {}
    for key in (
        "token",
        "atm_band_filter",
        "premium_enabled",
        "premium_min",
        "premium_max",
        "delta_enabled",
        "delta_min",
        "delta_max",
        "no_null_data",
    ):
        if key in raw:
            out[key] = raw.get(key)
    # Normalize premium_enabled from min/max when missing.
    if out.get("premium_min") is not None and out.get("premium_max") is not None:
        out["premium_enabled"] = bool(out.get("premium_enabled", True))
    return out


def master_filter_summary_label(master_filter: Mapping[str, Any] | None) -> str:
    """Short human label for logs / job messages."""
    mf = master_filter if isinstance(master_filter, Mapping) else {}
    if not mf:
        return "no training filter (full Master day)"
    parts: list[str] = []
    token = str(mf.get("token") or "").strip()
    if token:
        parts.append(f"token={token}")
    atm = mf.get("atm_band_filter")
    if atm is not None and str(atm).lower() != "all":
        try:
            parts.append(f"ATM ±{int(atm)}")
        except (TypeError, ValueError):
            parts.append(f"ATM={atm}")
    if mf.get("premium_enabled") or (
        mf.get("premium_min") is not None and mf.get("premium_max") is not None
    ):
        parts.append(f"LTP {mf.get('premium_min')}–{mf.get('premium_max')}")
    if mf.get("delta_enabled") and mf.get("delta_min") is not None and mf.get("delta_max") is not None:
        parts.append(f"|delta| {mf.get('delta_min')}–{mf.get('delta_max')}")
    if mf.get("no_null_data"):
        parts.append("no-null")
    return " · ".join(parts) if parts else "training filter"


def resolve_model_seen_trading_days(
    lab: Any,
    *,
    parent_trading_days: list[str] | tuple[str, ...] | set[str] | None = None,
) -> set[str]:
    """Trading days this model used for training / walk-forward validation.

    Prefer explicit day labels in lab snapshots. When those are missing or only
    say ``All days`` (no ISO dates), fall back to *parent_trading_days* — the
    parent training-export / registry catalog. Master-only extras stay Unseen.
    """
    out: set[str] = set()
    snap = getattr(lab, "dataset_snapshot", None) or {}
    if isinstance(snap, Mapping):
        _collect_from_mapping(out, snap)
        nested = snap.get("dataset_build_snapshot")
        if isinstance(nested, Mapping):
            _collect_from_mapping(out, nested)
            lineage = nested.get("dataset_lineage")
            if isinstance(lineage, Mapping):
                _collect_from_mapping(out, lineage)
    cfg = getattr(lab, "training_config_snapshot", None) or {}
    if isinstance(cfg, Mapping):
        _collect_from_mapping(out, cfg)
        split = cfg.get("split")
        if isinstance(split, Mapping):
            _collect_from_mapping(out, split)
            wf = split.get("walk_forward")
            if isinstance(wf, Mapping):
                _collect_from_mapping(out, wf)
    wf_snap = getattr(lab, "wf_snapshot", None) or {}
    if isinstance(wf_snap, Mapping):
        _collect_from_mapping(out, wf_snap)
        for fold in wf_snap.get("folds") or []:
            if isinstance(fold, Mapping):
                _collect_from_mapping(out, fold)
                for block_key in ("train", "validation", "val"):
                    block = fold.get(block_key)
                    if isinstance(block, Mapping):
                        _collect_from_mapping(out, block)
    if not out and parent_trading_days:
        out = {
            str(d).strip()
            for d in parent_trading_days
            if str(d or "").strip()
        }
    return out


def dataset_type_for_day(trading_day: str, seen_days: set[str] | None) -> str:
    """Classify one day. Empty / unknown seen set → Seen (legacy safe default)."""
    day = str(trading_day or "").strip()
    if not day:
        return DATASET_TYPE_SEEN
    if not seen_days:
        return DATASET_TYPE_SEEN
    return DATASET_TYPE_SEEN if day in seen_days else DATASET_TYPE_UNSEEN


def build_day_dataset_types(
    trading_days: list[str],
    seen_days: set[str] | None,
) -> dict[str, str]:
    return {
        str(d): dataset_type_for_day(str(d), seen_days)
        for d in trading_days
        if str(d or "").strip()
    }


def resolve_master_db_path_for_lab(
    lab: Any,
    *,
    data_dir: str,
    parent_meta: Mapping[str, Any] | None = None,
    extra_paths: list[str] | None = None,
) -> str | None:
    """Absolute Master DB path linked to this lab / parent dataset, if available."""
    candidates: list[str] = []
    snap = getattr(lab, "dataset_snapshot", None) or {}
    if isinstance(snap, Mapping):
        for key in ("master_db_path", "master_path"):
            raw = str(snap.get(key) or "").strip()
            if raw:
                candidates.append(raw)
        nested = snap.get("dataset_build_snapshot")
        if isinstance(nested, Mapping):
            raw = str(nested.get("master_db_path") or "").strip()
            if raw:
                candidates.append(raw)
    if isinstance(parent_meta, Mapping):
        raw = str(parent_meta.get("master_db_path") or "").strip()
        if raw:
            candidates.append(raw)
    for raw in extra_paths or []:
        text = str(raw or "").strip()
        if text:
            candidates.append(text)

    data_dir = str(data_dir or "").strip()
    for raw in candidates:
        if not raw:
            continue
        if os.path.isfile(raw):
            return os.path.abspath(raw)
        if data_dir:
            joined = os.path.normpath(os.path.join(data_dir, raw))
            if os.path.isfile(joined):
                return os.path.abspath(joined)
    return None


def load_master_day_row_counts(master_db_path: str) -> dict[str, int]:
    """``trading_day → row_count`` from Master Dataset metadata tables."""
    if not master_db_path or not os.path.isfile(master_db_path):
        return {}
    try:
        from chain_replay_ml.dataset_builder.master_store import MasterStore

        store = MasterStore(master_db_path)
        store.open()
        try:
            out: dict[str, int] = {}
            for row in store.read_master_days():
                day = str(row.get("trading_day") or "").strip()
                if not day:
                    continue
                try:
                    n = int(row.get("row_count") or 0)
                except (TypeError, ValueError):
                    n = 0
                out[day] = n
            if out:
                return out
            for day in store.distinct_trading_days():
                d = str(day or "").strip()
                if d:
                    out.setdefault(d, 0)
            return out
        finally:
            store.close()
    except Exception:
        return {}
