"""Shared time-shift helpers for Lag / Difference / Return transforms."""

from __future__ import annotations

from typing import Any

import pandas as pd


class LagConfigError(ValueError):
    """Invalid time-shift configuration (fail-fast, no silent skip/round)."""


def resolve_lag_row_offsets(
    lag_seconds: list[float | int],
    sample_interval_sec: float,
) -> list[tuple[float, int]]:
    """Map each lag (seconds) → exact integer row offset."""
    try:
        interval = float(sample_interval_sec)
    except (TypeError, ValueError) as exc:
        raise LagConfigError(
            "Lag Transformation\n"
            "sample_interval_sec is missing or invalid.\n"
            f"Got: {sample_interval_sec!r}"
        ) from exc
    if interval <= 0:
        raise LagConfigError(
            "Lag Transformation\n"
            "sample_interval_sec must be > 0.\n"
            f"Got: {interval}"
        )

    out: list[tuple[float, int]] = []
    seen: set[float] = set()
    for raw in lag_seconds:
        try:
            sec = float(raw)
        except (TypeError, ValueError) as exc:
            raise LagConfigError(
                "Lag Transformation\n"
                f"Invalid lag_seconds value: {raw!r}"
            ) from exc
        if sec <= 0:
            raise LagConfigError(
                "Lag Transformation\n"
                f"lag_seconds must be > 0.\n"
                f"Got: {sec}"
            )
        if sec in seen:
            continue
        seen.add(sec)
        rows = sec / interval
        rows_i = int(round(rows))
        if abs(rows - rows_i) > 1e-9 or rows_i < 1:
            raise LagConfigError(
                "Lag Transformation\n"
                "Invalid lag_seconds (not divisible by sample interval).\n"
                f"lag_seconds       : {sec}\n"
                f"sample_interval_sec: {interval}\n"
                f"rows              : {rows}\n"
                "Choose lag_seconds that are exact multiples of the sample interval."
            )
        out.append((sec, rows_i))
    return out


def resolve_transform_params(
    transform_id: str,
    instance_params: dict[str, Any] | None,
    context_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve params for a transform instance.

    Instance params always win when present. Config lookup by ``id`` is only a
    fallback for callers that did not stash params on the instance.

    Never merge the first matching config entry *over* instance params: analysis
    pipelines often have many stages with the same id (e.g. several Difference
    steps). Overwriting would make every stage reuse the first entry's features
    (and skip outputs such as ``ltp_step``).
    """
    if isinstance(instance_params, dict) and instance_params:
        return dict(instance_params)
    for entry in (context_config or {}).get("transformations") or []:
        if isinstance(entry, dict) and str(entry.get("id") or "") == transform_id:
            if isinstance(entry.get("params"), dict):
                return dict(entry["params"])
            break
    return dict(instance_params or {})


def partition_columns(params: dict[str, Any], df: pd.DataFrame) -> list[str]:
    raw = params.get("partition_by")
    if raw is None:
        raw = params.get("group_by")
    cols = [str(c).strip() for c in (raw or []) if str(c).strip()]
    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise LagConfigError(
                "Feature Transformation\n"
                "partition_by column not found\n"
                + "\n".join(missing)
            )
        return cols
    for cand in (("trading_day", "token"), ("token",), ("trading_day",)):
        if all(c in df.columns for c in cand):
            return list(cand)
    return []


def normalize_sample_interval_value(interval: Any) -> float | int | None:
    """Coerce a positive sample interval; integers stay ints for stable JSON."""
    try:
        if interval is None:
            return None
        value = float(interval)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return int(value) if value.is_integer() else value


def extract_sample_interval_from_config(config: dict[str, Any] | None) -> float | int | None:
    """Best-effort sample interval from transformation params (after persist)."""
    if not isinstance(config, dict):
        return None
    for entry in config.get("transformations") or []:
        if not isinstance(entry, dict):
            continue
        params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
        value = normalize_sample_interval_value(params.get("sample_interval_sec"))
        if value is not None:
            return value
    return None


def resolve_sample_interval(context: Any, params: dict[str, Any]) -> float:
    interval = getattr(context, "sample_interval_sec", None)
    if interval is None:
        interval = params.get("sample_interval_sec")
    if interval is None:
        dataset_info = getattr(context, "dataset_info", None) or {}
        metadata = getattr(context, "metadata", None) or {}
        interval = (
            dataset_info.get("sample_interval_sec")
            or metadata.get("sample_interval_sec")
            or (metadata.get("sampling") or {}).get("interval_sec")
            or (metadata.get("dataset_configuration") or {})
            .get("sampling", {})
            .get("interval_sec")
            or (metadata.get("dataset_configuration") or {}).get("sampling_interval_sec")
        )
    try:
        value = float(interval) if interval is not None else 0.0
    except (TypeError, ValueError):
        value = 0.0
    return value


def persist_sample_interval(
    context: Any,
    transform_id: str,
    interval: float,
) -> None:
    value = normalize_sample_interval_value(interval)
    if value is None:
        return
    for entry in (getattr(context, "config", None) or {}).get("transformations") or []:
        if not isinstance(entry, dict) or str(entry.get("id") or "") != transform_id:
            continue
        params = entry.get("params")
        if not isinstance(params, dict):
            params = {}
            entry["params"] = params
        params["sample_interval_sec"] = value
        break


def _normalize_horizon_column(column: Any) -> Any:
    """Keep string names and output→name dicts; drop empty values."""
    if column is None:
        return None
    if isinstance(column, dict):
        mapped = {
            str(k).strip(): str(v).strip()
            for k, v in column.items()
            if str(k).strip() and v is not None and str(v).strip()
        }
        return mapped or None
    text = str(column).strip()
    return text or None


def parse_features_and_horizons(
    *,
    transform_name: str,
    params: dict[str, Any],
    sample_interval_sec: float,
) -> tuple[list[str], list[tuple[float, int, str | None, Any]]]:
    """Return (features, [(seconds, row_offset, suffix, column), ...]).

    ``horizons`` entries may be seconds or dicts::
        {"seconds": 60, "suffix": "1m", "column": "oi_change_1m"}
        {"seconds": 300, "column": {"range_pos": "spot_range_pos_5m"}}
    """
    features = [str(f).strip() for f in (params.get("features") or []) if str(f).strip()]
    horizons_raw = params.get("horizons")
    lag_seconds_raw = params.get("lag_seconds")
    if lag_seconds_raw is None and params.get("horizons_sec") is not None:
        lag_seconds_raw = params.get("horizons_sec")
    if lag_seconds_raw is None and params.get("lags") is not None and horizons_raw is None:
        raise LagConfigError(
            f"{transform_name}\n"
            "Deprecated params.lags (row offsets) is no longer supported.\n"
            "Use params.lag_seconds or params.horizons."
        )

    structured: list[tuple[float, str | None, Any]] = []
    if isinstance(horizons_raw, list) and horizons_raw:
        for item in horizons_raw:
            if isinstance(item, dict):
                try:
                    sec = float(item.get("seconds"))
                except (TypeError, ValueError) as exc:
                    raise LagConfigError(
                        f"{transform_name}\nInvalid horizons.seconds: {item!r}"
                    ) from exc
                suffix = item.get("suffix")
                column = _normalize_horizon_column(item.get("column"))
                structured.append((
                    sec,
                    str(suffix).strip() if suffix is not None else None,
                    column,
                ))
            else:
                try:
                    structured.append((float(item), None, None))
                except (TypeError, ValueError) as exc:
                    raise LagConfigError(
                        f"{transform_name}\nInvalid horizons entry: {item!r}"
                    ) from exc
    else:
        for raw in list(lag_seconds_raw or []):
            try:
                structured.append((float(raw), None, None))
            except (TypeError, ValueError) as exc:
                raise LagConfigError(
                    f"{transform_name}\nInvalid lag_seconds value: {raw!r}"
                ) from exc

    if not features:
        raise LagConfigError(
            f"{transform_name}\n"
            "params.features is empty.\n"
            "Select at least one feature."
        )
    if not structured:
        # Derive from shared interval horizon policy only when horizons were omitted.
        omitted = horizons_raw is None and lag_seconds_raw is None
        if omitted:
            try:
                from .horizon_policy import default_horizons_for_interval

                for sec in default_horizons_for_interval(sample_interval_sec):
                    structured.append((float(sec), None, None))
            except Exception:
                structured = []
    if not structured:
        raise LagConfigError(
            f"{transform_name}\n"
            "params.lag_seconds / params.horizons is empty.\n"
            "Provide durations in seconds, or omit them to use horizon_policy.json "
            f"for sample_interval_sec={sample_interval_sec}."
        )

  # Preserve every horizon entry even when multiple columns share the same seconds
    # (e.g. oi_change_1m and oi_change_pct_1m both at 60s).
    out: list[tuple[float, int, str | None, Any]] = []
    for sec, suffix, column in structured:
        rows_pairs = resolve_lag_row_offsets([sec], sample_interval_sec)
        if not rows_pairs:
            continue
        _, rows = rows_pairs[0]
        out.append((sec, rows, suffix, column))
    return features, out


def shift_feature_columns(
    frame: pd.DataFrame,
    *,
    feature: str,
    rows: int,
    partition_by: list[str],
) -> pd.Series:
    if partition_by:
        return frame.groupby(partition_by, sort=False, group_keys=False)[feature].transform(
            lambda s, n=rows: s.shift(n)
        )
    return frame[feature].shift(rows)


def add_shifted_columns_via_polars(
    df: pd.DataFrame,
    *,
    specs: list[tuple[str, int, str]],
    partition_by: list[str],
) -> pd.DataFrame:
    """Compute lag/shift columns with Polars, return Pandas (P2 internal path).

    ``specs`` items are ``(source_feature, row_offset, output_column)``.
    Falls back to pandas ``shift_feature_columns`` if Polars is unavailable.
    """
    if not specs:
        return df
    try:
        from chain_replay_ml.frame_backend import (
            arrow_table_to_polars,
            polars_to_pandas,
            require_polars,
        )
    except Exception:
        return _add_shifted_columns_pandas(df, specs=specs, partition_by=partition_by)

    try:
        pl = require_polars()
    except ImportError:
        return _add_shifted_columns_pandas(df, specs=specs, partition_by=partition_by)

    # Prefer zero-copy when possible: pandas → arrow → polars
    try:
        import pyarrow as pa

        table = pa.Table.from_pandas(df, preserve_index=False)
        pl_df = arrow_table_to_polars(table)
    except Exception:
        pl_df = pl.from_pandas(df)

    exprs = []
    for feature, rows, out_col in specs:
        n = int(rows)
        if partition_by:
            expr = pl.col(feature).shift(n).over(partition_by).alias(out_col)
        else:
            expr = pl.col(feature).shift(n).alias(out_col)
        exprs.append(expr)
    pl_df = pl_df.with_columns(exprs)
    return polars_to_pandas(pl_df)


def _add_shifted_columns_pandas(
    df: pd.DataFrame,
    *,
    specs: list[tuple[str, int, str]],
    partition_by: list[str],
) -> pd.DataFrame:
    new_cols: dict[str, pd.Series] = {}
    for feature, rows, out_col in specs:
        new_cols[out_col] = shift_feature_columns(
            df, feature=feature, rows=int(rows), partition_by=partition_by
        )
    if not new_cols:
        return df
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
