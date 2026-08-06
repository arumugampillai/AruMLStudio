"""Apply parent-dataset Feature Transformations during prediction.

Prediction must recreate the parent dataset feature space before validating
model features. When a day is loaded from Master, the shared transformation
pipeline (Lag today; Difference / Return / Rolling / EMA later) is executed
from parent dataset metadata — never reimplemented in the prediction worker.
"""

from __future__ import annotations

from typing import Any

# Param keys that typically list source feature / partition columns.
_SOURCE_LIST_KEYS: tuple[str, ...] = (
    "features",
    "feature",  # singular alias (e.g. Rolling / OHLC)
    "columns",
    "inputs",
    "source_features",
    "left",
    "right",
    "partition_by",
    "group_by",
)


def _append_source_column(name: Any, *, seen: set[str], out: list[str]) -> None:
    col = str(name or "").strip()
    if col and col not in seen:
        seen.add(col)
        out.append(col)


def transformation_config_from_dataset_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize the parent dataset's saved transformation pipeline."""
    from chain_replay_ml.dataset_builder.transformations import (
        default_transformation_config,
        normalize_transformation_config,
    )

    if not isinstance(meta, dict) or not meta:
        return default_transformation_config()
    if isinstance(meta.get("transformations"), dict):
        return normalize_transformation_config(meta.get("transformations"))
    return normalize_transformation_config({
        "transformation_pipeline_version": meta.get("transformation_pipeline_version"),
        "transformations": meta.get("transformations"),
    })


def pipeline_has_enabled_transforms(config: dict[str, Any] | None) -> bool:
    from chain_replay_ml.dataset_builder.transformations import describe_pipeline

    if not config:
        return False
    return int(describe_pipeline(config).enabled) > 0


def sample_interval_sec_from_meta(
    meta: dict[str, Any] | None,
    *,
    fallback: float | int | None = None,
) -> float | None:
    """Resolve sample interval used for row-offset transforms and experiment identity.

    Preference order (experiment identity first):
      1. top-level ``sample_interval_sec``
      2. ``sampling.interval_sec``
      3. ``dataset_configuration.sampling_interval_sec`` / nested sampling
      4. ``interval_sec``
      5. each transform ``params.sample_interval_sec``
      6. ``fallback``
    """
    if isinstance(meta, dict):
        sampling = meta.get("sampling") if isinstance(meta.get("sampling"), dict) else {}
        ds_cfg = (
            meta.get("dataset_configuration")
            if isinstance(meta.get("dataset_configuration"), dict)
            else {}
        )
        ds_sampling = (
            ds_cfg.get("sampling") if isinstance(ds_cfg.get("sampling"), dict) else {}
        )
        for candidate in (
            meta.get("sample_interval_sec"),
            sampling.get("interval_sec"),
            ds_cfg.get("sampling_interval_sec"),
            ds_sampling.get("interval_sec"),
            meta.get("interval_sec"),
        ):
            try:
                if candidate is not None and float(candidate) > 0:
                    return float(candidate)
            except (TypeError, ValueError):
                continue
        for entry in meta.get("transformations") or []:
            if not isinstance(entry, dict):
                continue
            params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
            try:
                sec = params.get("sample_interval_sec")
                if sec is not None and float(sec) > 0:
                    return float(sec)
            except (TypeError, ValueError):
                continue
    if fallback is not None:
        try:
            if float(fallback) > 0:
                return float(fallback)
        except (TypeError, ValueError):
            pass
    return None


def source_columns_for_transformations(config: dict[str, Any] | None) -> list[str]:
    """Collect source / partition columns required by enabled transforms.

    Intentionally generic: reads known list-valued param keys so Lag and future
    transforms work without prediction-specific branching. Also walks nested
    Interaction ``pairs[].left/right`` and Derived/Anchor Return
    ``outputs[].feature`` (dict entries only — not OHLC field-name lists).
    """
    from chain_replay_ml.dataset_builder.transformations import normalize_transformation_config

    cfg = normalize_transformation_config(config)
    out: list[str] = []
    seen: set[str] = set()
    for entry in cfg.get("transformations") or []:
        if not isinstance(entry, dict) or not bool(entry.get("enabled", False)):
            continue
        params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
        for key in _SOURCE_LIST_KEYS:
            raw = params.get(key)
            values: list[Any]
            if isinstance(raw, (list, tuple, set)):
                values = list(raw)
            elif isinstance(raw, str) and raw.strip():
                values = [raw]
            else:
                continue
            for item in values:
                _append_source_column(item, seen=seen, out=out)
        # Interaction: pairs[].left / pairs[].right
        pairs = params.get("pairs")
        if isinstance(pairs, (list, tuple)):
            for pair in pairs:
                if not isinstance(pair, dict):
                    continue
                _append_source_column(pair.get("left"), seen=seen, out=out)
                _append_source_column(pair.get("right"), seen=seen, out=out)
        # Derived / Anchor Return: outputs[].feature (skip bare OHLC field lists)
        outputs = params.get("outputs")
        if isinstance(outputs, (list, tuple)):
            for item in outputs:
                if isinstance(item, dict) and "feature" in item:
                    _append_source_column(item.get("feature"), seen=seen, out=out)
    return out


def expand_columns_for_master_load(
    base_columns: list[str] | None,
    transformation_config: dict[str, Any] | None,
) -> list[str]:
    """Union identity/model columns with transform source columns for Master SELECT."""
    out: list[str] = []
    seen: set[str] = set()
    for col in list(base_columns or []) + source_columns_for_transformations(transformation_config):
        name = str(col or "").strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def apply_parent_dataset_transformations(
    frame: Any,
    *,
    transformation_config: dict[str, Any] | None,
    sample_interval_sec: float | int | None = None,
    data_dir: str | None = None,
    dataset_name: str | None = None,
    log_fn: Any | None = None,
) -> Any:
    """Run the shared Feature Transformation pipeline on a Master-loaded day frame.

    Returns the input frame unchanged when no transforms are enabled.
    """
    import pandas as pd

    from chain_replay_ml.dataset_builder.transformations import (
        normalize_transformation_config,
        run_transformation_pipeline,
    )
    from chain_replay_ml.dataset_builder.transformations.base import TransformContext

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame
    cfg = normalize_transformation_config(transformation_config)
    if not pipeline_has_enabled_transforms(cfg):
        return frame

    interval = None
    if sample_interval_sec is not None:
        try:
            interval = float(sample_interval_sec)
        except (TypeError, ValueError):
            interval = None
    if interval is None or interval <= 0:
        # Prefer interval stamped into lag params by Dataset Builder.
        for entry in cfg.get("transformations") or []:
            if not isinstance(entry, dict):
                continue
            params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
            try:
                sec = params.get("sample_interval_sec")
                if sec is not None and float(sec) > 0:
                    interval = float(sec)
                    break
            except (TypeError, ValueError):
                continue

    def _log(msg: str) -> None:
        if log_fn is not None:
            try:
                log_fn(str(msg))
            except Exception:
                pass

    ctx = TransformContext(
        config=cfg,
        data_dir=data_dir,
        dataset_name=dataset_name,
        sample_interval_sec=interval,
        dataset_info={
            "sample_interval_sec": interval,
        } if interval is not None else {},
        metadata={
            "sampling": {"interval_sec": interval},
        } if interval is not None else {},
        logger=_log,
    )
    pipe = run_transformation_pipeline(frame, cfg, context=ctx, log_fn=_log)
    return pipe.frame
