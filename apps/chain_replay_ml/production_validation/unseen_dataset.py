"""Resolve or create ``unseen_*`` Dataset Registry entries for Production Validation.

Seen = trading days used for the selected model's training / WF.
Unseen = Master Dataset trading days not in Seen.

Unseen datasets are built via the Feature Transformations analysis path
(``create_analysis_dataset`` with ``include_pipeline=True``) so Registry +
Pipeline Features match FT analysis datasets; name still starts with ``unseen_``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from .types import UnseenDatasetResolveResult

ProgressCb = Callable[[str, int, int], None]

_DAY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_HASH_LEN = 8
_SLUG_MAX = 40


def _safe_slug(text: str, *, max_len: int = _SLUG_MAX) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", str(text or "").strip()).strip("._-")
    cleaned = cleaned or "model"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("._-") or "model"
    return cleaned


def unseen_dataset_identity_hash(
    *,
    master_db_path: str,
    unseen_days: list[str] | tuple[str, ...] | set[str],
    master_filter: Mapping[str, Any] | None = None,
    parent_dataset: str | None = None,
    feature_project_id: str | None = None,
    pipeline_id: str | None = None,
    pipeline_snapshot_id: str | None = None,
    include_pipeline: bool = True,
    include_registry: bool = True,
) -> str:
    """Stable short hash for ``unseen_<slug>_<hash>`` naming / reuse."""
    norm_fpid = str(feature_project_id or "all").strip().lower() if feature_project_id is not None else None
    norm_pid = str(pipeline_id or "").strip().upper() if pipeline_id is not None else None
    norm_snap = str(pipeline_snapshot_id or "").strip() if pipeline_snapshot_id is not None else None
    payload = {
        "master_db_path": os.path.normpath(str(master_db_path or "")).replace("\\", "/").lower(),
        "unseen_days": sorted({str(d).strip() for d in unseen_days if str(d).strip()}),
        "master_filter": _stable_filter(master_filter),
        "parent_dataset": str(parent_dataset or "").strip() or None,
        "feature_project_id": norm_fpid,
        "pipeline_id": norm_pid,
        "pipeline_snapshot_id": norm_snap,
        "keep_pipeline_owned": True,
        "include_pipeline": bool(include_pipeline),
        "include_registry": bool(include_registry),
        "feature_sources": "registry+pipeline" if (include_pipeline and include_registry) else ("pipeline" if include_pipeline else "registry"),
        "dataset_kind": "unseen",
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:_HASH_LEN]


def build_unseen_dataset_name(
    *,
    model_name: str,
    identity_hash: str,
    parent_dataset: str | None = None,
) -> str:
    """Registry name: ``unseen_<model_or_parent>_<hash>``."""
    base = str(parent_dataset or "").strip() or str(model_name or "").strip() or "model"
    slug = _safe_slug(base)
    h = str(identity_hash or "").strip().lower()[:_HASH_LEN]
    if not h:
        h = "00000000"
    return f"unseen_{slug}_{h}"


def _stable_filter(master_filter: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(master_filter, Mapping) or not master_filter:
        return {}
    keys = (
        "token",
        "atm_band_filter",
        "premium_enabled",
        "premium_min",
        "premium_max",
        "delta_enabled",
        "delta_min",
        "delta_max",
        "no_null_data",
    )
    out: dict[str, Any] = {}
    for key in keys:
        if key in master_filter and master_filter.get(key) is not None:
            out[key] = master_filter.get(key)
    return out


def _load_json(path: str) -> dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: str, doc: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(doc), fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _package_seen_source(data_dir: str, model_name: str) -> SimpleNamespace:
    """Build a lab-like object for ``resolve_model_seen_trading_days``."""
    from chain_replay_ml.training.paths import model_artifact_paths, safe_model_name

    paths = model_artifact_paths(data_dir, safe_model_name(model_name))
    snap = _load_json(paths["dataset_build_snapshot_json"])
    cfg = _load_json(paths.get("config_json", ""))
    train_cfg = _load_json(os.path.join(paths["package_dir"], "training_config.json"))
    if not train_cfg:
        train_cfg = cfg
    # Prefer nested snapshot on config when standalone file is thin.
    nested = cfg.get("dataset_build_snapshot") if isinstance(cfg.get("dataset_build_snapshot"), dict) else {}
    if nested and (not snap or not snap.get("trading_day_labels")):
        merged = dict(nested)
        merged.update({k: v for k, v in snap.items() if v not in (None, "", [], {})})
        snap = merged
    meta = _load_json(os.path.join(paths["package_dir"], "metadata.json"))
    if isinstance(meta.get("dataset_build_snapshot"), dict) and not snap:
        snap = dict(meta["dataset_build_snapshot"])

    wf: dict[str, Any] = {}
    wf_dir = os.path.join(paths["package_dir"], "walk_forward")
    for fname in ("summary.json", "folds.json", "display.json"):
        part = _load_json(os.path.join(wf_dir, fname))
        if part:
            if fname == "folds.json" and isinstance(part.get("folds"), list):
                wf["folds"] = part["folds"]
            else:
                wf.update(part)
    # Some packages store WF under training_summary / metrics.
    summary = _load_json(os.path.join(paths["package_dir"], "training_summary.json"))
    if isinstance(summary.get("walk_forward"), dict) and not wf:
        wf = dict(summary["walk_forward"])

    return SimpleNamespace(
        dataset_snapshot=snap or {},
        training_config_snapshot=train_cfg or {},
        wf_snapshot=wf or {},
    )


def _parent_dataset_name(snap: Mapping[str, Any], train_cfg: Mapping[str, Any]) -> str | None:
    for src in (snap, train_cfg):
        if not isinstance(src, Mapping):
            continue
        for key in ("dataset_name", "dataset", "parent_dataset"):
            raw = str(src.get(key) or "").strip()
            if raw:
                return raw
        meta = src.get("dataset_metadata")
        if isinstance(meta, Mapping):
            raw = str(meta.get("dataset_name") or meta.get("dataset") or "").strip()
            if raw:
                return raw
    return None


def _parent_trading_days(data_dir: str, parent_dataset: str | None) -> list[str]:
    if not parent_dataset:
        return []
    try:
        from chain_replay_ml.dataset_builder.append_ops import load_dataset_metadata

        meta, _paths = load_dataset_metadata(data_dir, parent_dataset)
    except Exception:
        return []
    days: list[str] = []
    for block_key in ("days", "sources"):
        block = meta.get(block_key)
        if isinstance(block, list):
            for item in block:
                if isinstance(item, Mapping):
                    d = str(item.get("trading_day") or item.get("day") or "").strip()
                else:
                    d = str(item or "").strip()
                if _DAY_RE.fullmatch(d):
                    days.append(d)
    labels = str(meta.get("trading_day_labels") or "").strip()
    if labels and labels != "—":
        for part in labels.split(","):
            d = part.strip()
            if _DAY_RE.fullmatch(d):
                days.append(d)
    return sorted(set(days))


def _master_days_and_path(
    data_dir: str,
    lab_like: Any,
    *,
    parent_meta: Mapping[str, Any] | None,
) -> tuple[str | None, list[str]]:
    from chain_replay_ml.model_lab.prediction_dataset_type import (
        load_master_day_row_counts,
        resolve_master_db_path_for_lab,
    )

    master_path = resolve_master_db_path_for_lab(
        lab_like,
        data_dir=data_dir,
        parent_meta=parent_meta,
    )
    if not master_path:
        # Fall back to snapshot / parent meta absolute or relative paths already tried.
        snap = getattr(lab_like, "dataset_snapshot", None) or {}
        if isinstance(snap, Mapping):
            for key in ("master_db_path", "master_path"):
                raw = str(snap.get(key) or "").strip()
                if raw and os.path.isfile(raw):
                    master_path = os.path.abspath(raw)
                    break
                if raw and data_dir:
                    joined = os.path.normpath(os.path.join(data_dir, raw))
                    if os.path.isfile(joined):
                        master_path = os.path.abspath(joined)
                        break
    if not master_path or not os.path.isfile(master_path):
        return None, []
    rows = load_master_day_row_counts(master_path)
    days = sorted(str(d).strip() for d in rows if str(d).strip() and _DAY_RE.fullmatch(str(d).strip()))
    return master_path, days


def _days_from_meta(meta: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    pv = meta.get("production_validation")
    if isinstance(pv, Mapping):
        for d in pv.get("unseen_days") or []:
            text = str(d).strip()
            if _DAY_RE.fullmatch(text):
                out.append(text)
    for block_key in ("days", "sources"):
        block = meta.get(block_key)
        if isinstance(block, list):
            for item in block:
                if isinstance(item, Mapping):
                    d = str(item.get("trading_day") or item.get("day") or "").strip()
                else:
                    d = str(item or "").strip()
                if _DAY_RE.fullmatch(d):
                    out.append(d)
    return sorted(set(out))


def _column_names_from_meta_and_parquet(
    meta: Mapping[str, Any],
    parquet_path: str | None,
) -> set[str]:
    cols: set[str] = set()
    for key in ("feature_columns", "transformed_feature_columns"):
        for raw in meta.get(key) or []:
            name = str(raw or "").strip()
            if name:
                cols.add(name)
    exec_block = meta.get("execution") if isinstance(meta.get("execution"), dict) else {}
    summary = exec_block.get("transformation_summary") if isinstance(exec_block, dict) else None
    if isinstance(summary, Mapping):
        for raw in summary.get("output_columns") or summary.get("added_columns") or []:
            name = str(raw or "").strip()
            if name:
                cols.add(name)
    if parquet_path and os.path.isfile(parquet_path):
        try:
            import pyarrow.parquet as pq

            cols.update(str(n) for n in pq.read_schema(parquet_path).names if str(n).strip())
        except Exception:
            pass
    return cols


def _has_pipeline_features(
    meta: Mapping[str, Any],
    *,
    data_dir: str,
    parquet_path: str | None = None,
) -> bool:
    """True when the dataset includes regenerated Pipeline Features (FT parity)."""
    try:
        from chain_replay_ml.dataset_builder.feature_sources_catalog import (
            pipeline_feature_names,
        )

        pipe_names = pipeline_feature_names(data_dir=data_dir)
    except Exception:
        pipe_names = []
    if not pipe_names:
        # No catalogue to require — accept keep_pipeline_owned builds.
        return bool(meta.get("keep_pipeline_owned"))
    cols = _column_names_from_meta_and_parquet(meta, parquet_path)
    present = sum(1 for n in pipe_names if n in cols)
    # Registry-only (~206) must not pass. Require a meaningful pipeline share.
    minimum = max(10, min(50, len(pipe_names) // 10))
    return present >= minimum


def _remove_dataset_files(data_dir: str, dataset_name: str) -> None:
    """Delete parquet/json so recreate can reuse the exact ``unseen_*`` name."""
    from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir

    safe = _safe_filename(dataset_name)
    out_dir = datasets_dir(data_dir)
    for ext in (".parquet", ".json"):
        path = os.path.join(out_dir, f"{safe}{ext}")
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass


def _existing_unseen_valid(
    *,
    data_dir: str,
    dataset_name: str,
    expected_days: list[str],
    identity_hash: str,
    expected_feature_project_id: str | None = None,
    expected_pipeline_id: str | None = None,
    expected_pipeline_snapshot_id: str | None = None,
    expected_include_pipeline: bool = True,
    expected_include_registry: bool = True,
) -> dict[str, Any] | None:
    from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir

    safe = _safe_filename(dataset_name)
    out_dir = datasets_dir(data_dir)
    json_path = os.path.join(out_dir, f"{safe}.json")
    parquet_path = os.path.join(out_dir, f"{safe}.parquet")
    if not os.path.isfile(json_path) or not os.path.isfile(parquet_path):
        return None
    meta = _load_json(json_path)
    if not meta:
        return None
    kind = str(meta.get("dataset_kind") or "").strip().lower()
    pv = meta.get("production_validation") if isinstance(meta.get("production_validation"), dict) else {}
    role = str(pv.get("role") or "").strip().lower()
    if kind != "unseen" and role != "unseen":
        return None
    stored_hash = str(pv.get("identity_hash") or "").strip().lower()
    if stored_hash and stored_hash != str(identity_hash).lower():
        return None
    have = set(_days_from_meta(meta))
    want = set(expected_days)
    if have != want:
        return None

    # Lineage parity checks:
    # 1. Feature project
    if expected_feature_project_id is not None:
        stored_fpid = str(meta.get("feature_project_id") or pv.get("feature_project_id") or "all").strip().lower()
        target_fpid = str(expected_feature_project_id or "all").strip().lower()
        if stored_fpid != target_fpid:
            return None

    # 2. Pipeline ID
    if expected_pipeline_id is not None:
        stored_pid = str(meta.get("pipeline_id") or pv.get("pipeline_id") or "").strip().upper()
        target_pid = str(expected_pipeline_id or "").strip().upper()
        if stored_pid != target_pid:
            return None

    # 3. Pipeline Snapshot ID
    if expected_pipeline_snapshot_id is not None:
        stored_snap = str(meta.get("pipeline_snapshot_id") or pv.get("pipeline_snapshot_id") or "").strip()
        target_snap = str(expected_pipeline_snapshot_id or "").strip()
        if target_snap and stored_snap and stored_snap != target_snap:
            return None

    if expected_include_pipeline:
        if not _has_pipeline_features(meta, data_dir=data_dir, parquet_path=parquet_path):
            return None

    return {
        "dataset_name": str(meta.get("dataset_name") or safe),
        "json_path": json_path,
        "parquet_path": parquet_path,
        "meta": meta,
    }


def _stamp_unseen_metadata(
    json_path: str,
    *,
    model_name: str,
    identity_hash: str,
    seen_days: list[str],
    unseen_days: list[str],
    parent_dataset: str | None,
    feature_project_id: str | None = None,
    pipeline_id: str | None = None,
    pipeline_snapshot_id: str | None = None,
    include_pipeline: bool = True,
    include_registry: bool = True,
) -> None:
    meta = _load_json(json_path)
    if not meta:
        return
    meta["dataset_kind"] = "unseen"
    meta["keep_pipeline_owned"] = True
    meta["production_validation"] = {
        "role": "unseen",
        "version": "1.0",
        "model_name": model_name,
        "parent_dataset": parent_dataset,
        "identity_hash": identity_hash,
        "feature_project_id": feature_project_id,
        "pipeline_id": pipeline_id,
        "pipeline_snapshot_id": pipeline_snapshot_id,
        "seen_days": list(seen_days),
        "unseen_days": list(unseen_days),
        "seen_day_count": len(seen_days),
        "unseen_day_count": len(unseen_days),
        "include_pipeline": include_pipeline,
        "include_registry": include_registry,
        "feature_sources": "registry+pipeline" if (include_pipeline and include_registry) else ("pipeline" if include_pipeline else "registry"),
        "compute_note": "compute coming",
    }
    _write_json(json_path, meta)


def _persist_package_status(
    data_dir: str,
    model_name: str,
    result: UnseenDatasetResolveResult,
) -> None:
    from chain_replay_ml.training.paths import model_package_dir, safe_model_name

    pkg = model_package_dir(data_dir, safe_model_name(model_name))
    path = os.path.join(pkg, "production_validation", "unseen_dataset.json")
    _write_json(path, result.as_dict())


def load_unseen_dataset_status(data_dir: str, model_name: str) -> dict[str, Any] | None:
    """Load last Phase A resolve status from the model package, if any."""
    from chain_replay_ml.training.paths import model_package_dir, safe_model_name

    pkg = model_package_dir(data_dir, safe_model_name(model_name))
    path = os.path.join(pkg, "production_validation", "unseen_dataset.json")
    doc = _load_json(path)
    return doc or None


def resolve_unseen_dataset(
    *,
    data_dir: str,
    model_name: str,
    create_if_missing: bool = True,
    on_progress: ProgressCb | None = None,
) -> UnseenDatasetResolveResult:
    """Resolve Seen/Unseen days and reuse or create ``unseen_*`` analysis dataset.

    Creation uses ``create_analysis_dataset`` (Registry + Pipeline Features),
    same as Feature Transformations Auto builds, with ``dataset_kind="unseen"``.
    Existing ``unseen_*`` entries are reused only when they still match identity
    and include pipeline features; incomplete registry-only artifacts are
    invalidated and rebuilt.
    """
    from chain_replay_ml.model_lab.prediction_dataset_type import (
        resolve_model_master_filter,
        resolve_model_seen_trading_days,
    )
    from chain_replay_ml.training.paths import safe_model_name

    name = str(model_name or "").strip()
    if not name:
        return UnseenDatasetResolveResult(
            ok=False,
            model_name="",
            error="model_name is required",
            status="error",
            message="Select a model first.",
        )

    safe = safe_model_name(name)
    lab_like = _package_seen_source(data_dir, safe)
    snap = getattr(lab_like, "dataset_snapshot", None) or {}
    train_cfg = getattr(lab_like, "training_config_snapshot", None) or {}
    parent_dataset = _parent_dataset_name(snap, train_cfg)

    parent_meta: dict[str, Any] = {}
    if parent_dataset:
        try:
            from chain_replay_ml.dataset_builder.append_ops import load_dataset_metadata

            parent_meta, _ = load_dataset_metadata(data_dir, parent_dataset)
        except Exception:
            parent_meta = {}

    parent_days = _parent_trading_days(data_dir, parent_dataset)
    seen = resolve_model_seen_trading_days(
        lab_like,
        parent_trading_days=parent_days or None,
    )
    seen_days = sorted(seen)

    master_path, master_days = _master_days_and_path(
        data_dir,
        lab_like,
        parent_meta=parent_meta or None,
    )
    if not master_path:
        result = UnseenDatasetResolveResult(
            ok=False,
            model_name=name,
            seen_days=seen_days,
            error="Master Dataset path not found for this model",
            status="error",
            message="Cannot resolve Master Dataset — link master_db_path on the training snapshot.",
        )
        _persist_package_status(data_dir, name, result)
        return result

    # Extract lineage with precedence: parent_meta -> train_cfg
    pipeline_id = parent_meta.get("pipeline_id") or train_cfg.get("pipeline_id")
    feature_project_id = (
        parent_meta.get("feature_project_id")
        or train_cfg.get("feature_project_id")
    )
    pipeline_snapshot_id = (
        parent_meta.get("pipeline_snapshot_id")
        or train_cfg.get("pipeline_snapshot_id")
    )
    if "include_pipeline" in parent_meta:
        include_pipeline = bool(parent_meta.get("include_pipeline"))
    elif "include_pipeline" in train_cfg:
        include_pipeline = bool(train_cfg.get("include_pipeline"))
    else:
        include_pipeline = True

    if "include_registry" in parent_meta:
        include_registry = bool(parent_meta.get("include_registry"))
    elif "include_registry" in train_cfg:
        include_registry = bool(train_cfg.get("include_registry"))
    else:
        include_registry = True

    unseen_days = sorted(d for d in master_days if d not in seen)
    master_filter = resolve_model_master_filter(lab_like, parent_meta=parent_meta or None)
    identity = unseen_dataset_identity_hash(
        master_db_path=master_path,
        unseen_days=unseen_days,
        master_filter=master_filter,
        parent_dataset=parent_dataset,
        feature_project_id=feature_project_id,
        pipeline_id=pipeline_id,
        pipeline_snapshot_id=pipeline_snapshot_id,
        include_pipeline=include_pipeline,
        include_registry=include_registry,
    )
    dataset_name = build_unseen_dataset_name(
        model_name=safe,
        identity_hash=identity,
        parent_dataset=parent_dataset,
    )

    if not unseen_days:
        result = UnseenDatasetResolveResult(
            ok=True,
            model_name=name,
            dataset_name=None,
            seen_days=seen_days,
            unseen_days=[],
            master_db_path=master_path,
            identity_hash=identity,
            status="empty",
            message=(
                f"No unseen days — Master has {len(master_days)} day(s), "
                f"all covered by Seen ({len(seen_days)})."
            ),
            compute_note="compute coming",
        )
        _persist_package_status(data_dir, name, result)
        return result

    existing = _existing_unseen_valid(
        data_dir=data_dir,
        dataset_name=dataset_name,
        expected_days=unseen_days,
        identity_hash=identity,
        expected_feature_project_id=feature_project_id,
        expected_pipeline_id=pipeline_id,
        expected_pipeline_snapshot_id=pipeline_snapshot_id,
        expected_include_pipeline=include_pipeline,
        expected_include_registry=include_registry,
    )
    if existing:
        result = UnseenDatasetResolveResult(
            ok=True,
            model_name=name,
            dataset_name=existing["dataset_name"],
            parquet_path=existing["parquet_path"],
            json_path=existing["json_path"],
            reused=True,
            created=False,
            seen_days=seen_days,
            unseen_days=unseen_days,
            master_db_path=master_path,
            identity_hash=identity,
            status="ready",
            message=f"Reused registry dataset {existing['dataset_name']} ({len(unseen_days)} unseen days).",
            compute_note="compute coming",
        )
        _persist_package_status(data_dir, name, result)
        return result

    if not create_if_missing:
        result = UnseenDatasetResolveResult(
            ok=True,
            model_name=name,
            dataset_name=dataset_name,
            seen_days=seen_days,
            unseen_days=unseen_days,
            master_db_path=master_path,
            identity_hash=identity,
            status="pending",
            message=f"Unseen dataset {dataset_name} not registered yet ({len(unseen_days)} days).",
            compute_note="compute coming",
        )
        _persist_package_status(data_dir, name, result)
        return result

    def _progress(msg: str, cur: int = 0, tot: int = 0, **_detail: Any) -> None:
        if on_progress is not None:
            try:
                on_progress(str(msg), int(cur or 0), int(tot or 0))
            except Exception:
                pass

    # Infer market / interval from Master meta or parent.
    market = "NIFTY"
    interval_sec = 3
    try:
        from chain_replay_ml.dataset_builder.master_store import MasterStore

        store = MasterStore(master_path)
        store.open()
        try:
            mc = store.get_meta("master_config") or {}
        finally:
            store.close()
        if isinstance(mc, dict):
            market = str(mc.get("market") or market).upper()
            interval_sec = int(mc.get("sampling_interval_sec") or mc.get("interval_sec") or interval_sec)
    except Exception:
        pass
    if parent_meta:
        market = str(parent_meta.get("market") or market).upper()
        sampling = parent_meta.get("sampling") if isinstance(parent_meta.get("sampling"), dict) else {}
        if sampling.get("interval_sec"):
            interval_sec = int(sampling["interval_sec"])
        elif parent_meta.get("sample_interval_sec"):
            interval_sec = int(parent_meta["sample_interval_sec"])

    filter_kwargs: dict[str, Any] = {}
    if master_filter:
        if master_filter.get("token"):
            filter_kwargs["token"] = master_filter.get("token")
        if master_filter.get("atm_band_filter") is not None:
            try:
                filter_kwargs["atm_band_filter"] = int(master_filter["atm_band_filter"])
            except (TypeError, ValueError):
                pass
        if master_filter.get("premium_enabled") or (
            master_filter.get("premium_min") is not None and master_filter.get("premium_max") is not None
        ):
            filter_kwargs["premium_enabled"] = True
            filter_kwargs["premium_min"] = master_filter.get("premium_min")
            filter_kwargs["premium_max"] = master_filter.get("premium_max")
        if master_filter.get("delta_enabled"):
            filter_kwargs["delta_enabled"] = True
            filter_kwargs["delta_min"] = master_filter.get("delta_min")
            filter_kwargs["delta_max"] = master_filter.get("delta_max")
        if master_filter.get("no_null_data"):
            filter_kwargs["no_null_data"] = bool(master_filter.get("no_null_data"))

    trading_day_filter = {
        "mode": "selected",
        "selected_days": len(unseen_days),
        "exported_days": len(unseen_days),
        "selected_dates": list(unseen_days),
        "exported_dates": list(unseen_days),
        "excluded_dates": list(seen_days),
    }

    # Incomplete registry-only artifacts must not block the exact unseen_* name.
    from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir

    safe_ds = _safe_filename(dataset_name)
    out_dir = datasets_dir(data_dir)
    stale_json = os.path.join(out_dir, f"{safe_ds}.json")
    stale_pq = os.path.join(out_dir, f"{safe_ds}.parquet")
    if os.path.isfile(stale_json) or os.path.isfile(stale_pq):
        stale_meta = _load_json(stale_json) if os.path.isfile(stale_json) else {}
        if not _has_pipeline_features(
            stale_meta,
            data_dir=data_dir,
            parquet_path=stale_pq if os.path.isfile(stale_pq) else None,
        ):
            _progress(
                f"Invalidating incomplete {dataset_name} (missing pipeline features)…",
                0,
                len(unseen_days),
            )
            _remove_dataset_files(data_dir, dataset_name)

    _progress(
        f"Production Validation: creating {dataset_name} (FT analysis path + pipeline)…",
        0,
        len(unseen_days),
    )

    def _analysis_progress(payload: Mapping[str, Any] | dict[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            return
        msg = str(payload.get("message") or "")
        cur = int(payload.get("overall_done") or payload.get("export_current") or 0)
        tot = int(payload.get("overall_total") or payload.get("export_total") or 0)
        _progress(msg, cur, tot)

    try:
        from chain_replay_ml.dataset_builder.analysis_dataset_export import (
            create_analysis_dataset,
        )
        from chain_replay_ml.dataset_builder.master_registry_export import (
            MasterRegistryExportError,
        )

        payload = create_analysis_dataset(
            data_dir,
            market=market,
            interval_sec=interval_sec,
            include_registry=include_registry,
            include_pipeline=include_pipeline,
            pipeline_id=pipeline_id,
            feature_project_id=feature_project_id,
            all_days=False,
            selected_days=unseen_days,
            master_db_path=master_path,
            dataset_name=dataset_name,
            dataset_kind="unseen",
            trading_day_filter=trading_day_filter,
            on_progress=_analysis_progress,
            **filter_kwargs,
        )
    except MasterRegistryExportError as exc:
        result = UnseenDatasetResolveResult(
            ok=False,
            model_name=name,
            dataset_name=dataset_name,
            seen_days=seen_days,
            unseen_days=unseen_days,
            master_db_path=master_path,
            identity_hash=identity,
            status="error",
            error=str(getattr(exc, "detail", None) or exc),
            message="Failed to create unseen analysis dataset.",
        )
        _persist_package_status(data_dir, name, result)
        return result
    except Exception as exc:
        result = UnseenDatasetResolveResult(
            ok=False,
            model_name=name,
            dataset_name=dataset_name,
            seen_days=seen_days,
            unseen_days=unseen_days,
            master_db_path=master_path,
            identity_hash=identity,
            status="error",
            error=str(exc),
            message="Failed to create unseen analysis dataset.",
        )
        _persist_package_status(data_dir, name, result)
        return result

    # create_analysis_dataset / export may suffix _2 if a stale file blocked the name.
    actual_name = str(payload.get("dataset_name") or dataset_name)
    json_path = str(payload.get("json_path") or "")
    parquet_path = str(payload.get("parquet_path") or "")
    if json_path:
        _stamp_unseen_metadata(
            json_path,
            model_name=name,
            identity_hash=identity,
            seen_days=seen_days,
            unseen_days=unseen_days,
            parent_dataset=parent_dataset,
            feature_project_id=feature_project_id,
            pipeline_id=pipeline_id,
            pipeline_snapshot_id=pipeline_snapshot_id,
            include_pipeline=include_pipeline,
            include_registry=include_registry,
        )

    # Best-effort Analysis Lab registration (Dataset Registry UI also scans JSON).
    if parquet_path and os.path.isfile(parquet_path):
        try:
            from chain_replay_ml.dataset_builder.analysis_lab_store import register_dataset

            register_dataset(data_dir, parquet_path, name=actual_name)
        except Exception:
            pass

    pipe_present = int(payload.get("pipeline_present") or 0)
    feat_count = int(payload.get("feature_count") or 0)
    result = UnseenDatasetResolveResult(
        ok=True,
        model_name=name,
        dataset_name=actual_name,
        parquet_path=parquet_path or None,
        json_path=json_path or None,
        reused=False,
        created=True,
        seen_days=seen_days,
        unseen_days=unseen_days,
        master_db_path=master_path,
        identity_hash=identity,
        status="ready",
        message=(
            f"Created unseen dataset {actual_name} "
            f"({len(unseen_days)} days, {feat_count} features"
            f"{f', {pipe_present} pipeline' if pipe_present else ''})."
        ),
        compute_note="compute coming",
    )
    _persist_package_status(data_dir, name, result)
    return result
