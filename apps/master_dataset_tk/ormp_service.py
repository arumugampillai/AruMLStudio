"""ORMP builds catalog + runner for ML Research Studio (Phase 2).

Does not touch Master Dataset. Builds land under ``ormp/outputs/``.
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

ORMP_VERSION = "0.1.0"

_BUILD_NAME_RE = re.compile(
    r"^ormp_dataset_bs(?P<bs>.+)_(?P<source>close|hlc3|ohlc4|typical_price)_"
    r"(?P<path>snapshot|continuous)(?:_(?P<ver>.+))?\.db$",
    re.IGNORECASE,
)


def _repo_root_from_chart_dir(chart_dir: str) -> str:
    chart_dir = os.path.abspath(chart_dir or "")
    parent = os.path.dirname(chart_dir)
    base = os.path.basename(chart_dir).lower()
    if base in ("apps", "chart"):
        return parent if base == "apps" else os.path.dirname(parent)
    grand = os.path.dirname(parent)
    # Fallback: walk up for ormp/
    cur = chart_dir
    for _ in range(5):
        if os.path.isdir(os.path.join(cur, "ormp")):
            return cur
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    return grand or chart_dir


def ensure_ormp_importable(chart_dir: str) -> str:
    """Add repo root to sys.path; return repo root."""
    root = _repo_root_from_chart_dir(chart_dir)
    if root and root not in sys.path:
        sys.path.insert(0, root)
    return root


def ormp_outputs_dir(chart_dir: str) -> str:
    root = ensure_ormp_importable(chart_dir)
    return os.path.join(root, "ormp", "outputs")


def default_candle_db(chart_dir: str) -> str:
    return os.path.join(os.path.abspath(chart_dir), "data", "angel_historic_bars.db")


def candle_coverage_dates(chart_dir: str) -> tuple[str | None, str | None]:
    """Min/max trading days available in 1m NIFTY historic bars (YYYY-MM-DD)."""
    db_path = default_candle_db(chart_dir)
    if not os.path.isfile(db_path):
        return None, None
    ensure_ormp_importable(chart_dir)
    from ormp.config import DEFAULT_INTERVAL_SEC, DEFAULT_NIFTY_TOKEN

    sql = """
        SELECT
            MIN(date(bucket_start, 'unixepoch', '+5 hours', '+30 minutes')),
            MAX(date(bucket_start, 'unixepoch', '+5 hours', '+30 minutes'))
        FROM angel_historic_bars
        WHERE token = ? AND interval_sec = ?
    """
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                sql, (str(DEFAULT_NIFTY_TOKEN), int(DEFAULT_INTERVAL_SEC))
            ).fetchone()
    except sqlite3.Error:
        return None, None
    if not row or not row[0] or not row[1]:
        return None, None
    return str(row[0]), str(row[1])


@dataclass
class OrmpBuildInfo:
    build_id: str
    path: str
    band_size_pct: float | None
    price_source: str
    path_mode: str
    version_tag: str = ""
    rows: int | None = None
    days: int | None = None
    from_date: str | None = None
    to_date: str | None = None
    built_at: float | None = None
    file_size_bytes: int | None = None
    status: str = "ready"
    ormp_version: str = ORMP_VERSION

    @property
    def display_name(self) -> str:
        bs = self.band_size_pct
        bs_s = f"{bs:g}" if bs is not None else "?"
        src = self.price_source.replace("_", " ").title().replace(" ", "")
        path = self.path_mode.title()
        base = f"BS{bs_s}_{src}_{path}"
        if self.version_tag:
            return f"{base} ({self.version_tag})"
        return base

    @property
    def built_at_label(self) -> str:
        if not self.built_at:
            return "—"
        try:
            return datetime.fromtimestamp(self.built_at).strftime("%Y-%m-%d %H:%M")
        except (OSError, OverflowError, ValueError):
            return "—"


def _parse_build_filename(name: str) -> dict[str, Any] | None:
    m = _BUILD_NAME_RE.match(name)
    if not m:
        return None
    bs_raw = m.group("bs").replace("p", ".")
    try:
        band_size = float(bs_raw)
    except ValueError:
        band_size = None
    return {
        "band_size_pct": band_size,
        "price_source": m.group("source").lower(),
        "path_mode": m.group("path").lower(),
        "version_tag": (m.group("ver") or "").strip(),
    }


def _read_build_meta(db_path: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows": None,
        "days": None,
        "from_date": None,
        "to_date": None,
    }
    try:
        with sqlite3.connect(db_path) as conn:
            try:
                out["rows"] = int(
                    conn.execute("SELECT COUNT(*) FROM ormp_samples").fetchone()[0]
                )
            except sqlite3.Error:
                pass
            try:
                row = conn.execute(
                    "SELECT COUNT(*), MIN(trading_day), MAX(trading_day) FROM ormp_day_summary"
                ).fetchone()
                if row:
                    out["days"] = int(row[0] or 0)
                    out["from_date"] = row[1]
                    out["to_date"] = row[2]
            except sqlite3.Error:
                pass
    except sqlite3.Error:
        pass
    return out


def list_ormp_builds(chart_dir: str) -> list[OrmpBuildInfo]:
    out_dir = ormp_outputs_dir(chart_dir)
    if not os.path.isdir(out_dir):
        return []
    builds: list[OrmpBuildInfo] = []
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".db") or not name.startswith("ormp_dataset_"):
            continue
        parsed = _parse_build_filename(name)
        if not parsed:
            continue
        path = os.path.join(out_dir, name)
        try:
            st = os.stat(path)
            size = int(st.st_size)
            mtime = float(st.st_mtime)
        except OSError:
            size = None
            mtime = None
        meta = _read_build_meta(path)
        builds.append(
            OrmpBuildInfo(
                build_id=os.path.splitext(name)[0],
                path=path,
                band_size_pct=parsed["band_size_pct"],
                price_source=parsed["price_source"],
                path_mode=parsed["path_mode"],
                version_tag=parsed.get("version_tag") or "",
                rows=meta["rows"],
                days=meta["days"],
                from_date=meta["from_date"],
                to_date=meta["to_date"],
                built_at=mtime,
                file_size_bytes=size,
                status="ready",
            )
        )
    builds.sort(key=lambda b: b.built_at or 0.0, reverse=True)
    return builds


def overview_snapshot(chart_dir: str, *, build_id: str | None = None) -> dict[str, Any]:
    builds = list_ormp_builds(chart_dir)
    selected = None
    if build_id:
        selected = next((b for b in builds if b.build_id == build_id), None)
    if selected is None and builds:
        selected = builds[0]
    candle_db = default_candle_db(chart_dir)
    cov_from, cov_to = candle_coverage_dates(chart_dir)
    return {
        "ormp_version": ORMP_VERSION,
        "candle_db_path": candle_db,
        "candle_db_exists": os.path.isfile(candle_db),
        "candle_from_date": cov_from,
        "candle_to_date": cov_to,
        "outputs_dir": ormp_outputs_dir(chart_dir),
        "builds": builds,
        "selected": selected,
        "build_count": len(builds),
    }


ProgressCb = Callable[[str, dict[str, Any]], None]


def run_ormp_build(
    chart_dir: str,
    *,
    band_size_pct: float,
    price_source: str,
    path_mode: str,
    from_date: str | None = None,
    to_date: str | None = None,
    on_progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Build an immutable ORMP artifact into ormp/outputs/."""
    ensure_ormp_importable(chart_dir)
    from ormp.config import OrmpConfig
    from ormp.dataset_builder import build_ormp_dataset
    from ormp.validate import run_unit_sanity, summarize_dataset

    run_unit_sanity()
    # Immutable versioned artifact (UI always stamps; CLI can omit suffix).
    artifact_suffix = datetime.now().strftime("v%Y%m%d_%H%M%S")
    cfg = OrmpConfig(
        candle_db_path=default_candle_db(chart_dir),
        output_dir=ormp_outputs_dir(chart_dir),
        band_size_pct=float(band_size_pct),
        price_source=price_source,  # type: ignore[arg-type]
        path_mode=path_mode,  # type: ignore[arg-type]
        from_date=from_date or None,
        to_date=to_date or None,
        artifact_suffix=artifact_suffix,
    )
    t0 = time.perf_counter()
    result = build_ormp_dataset(cfg, on_progress=on_progress)
    summary = summarize_dataset(result["output_path"])
    return {
        "ok": bool(result.get("ok") and summary.get("ok")),
        "build": result,
        "summary": summary,
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "ormp_version": ORMP_VERSION,
    }


def suggest_ormp_dataset_name(
    build: OrmpBuildInfo,
    *,
    horizon_min: int,
    label_type: str,
    chart_dir: str = "",
) -> str:
    if chart_dir:
        ensure_ormp_importable(chart_dir)
    else:
        ensure_ormp_importable(os.path.dirname(os.path.abspath(build.path)))
    from ormp.training_export import suggest_dataset_name

    return suggest_dataset_name(
        band_size_pct=build.band_size_pct,
        price_source=build.price_source,
        path_mode=build.path_mode,
        horizon_min=int(horizon_min),
        label_type=label_type,  # type: ignore[arg-type]
    )


def export_ormp_training_dataset(
    chart_dir: str,
    *,
    build: OrmpBuildInfo,
    dataset_name: str,
    feature_columns: list[str],
    label_type: str,
    horizon_min: int,
    from_date: str | None,
    to_date: str | None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Slice ORMP build → labeled Parquet registered in Dataset Registry."""
    ensure_ormp_importable(chart_dir)
    from ormp.training_export import export_training_frame

    from chain_replay_ml.dataset_builder.writer import write_dataset

    from .build_service import chart_data_dir

    if on_progress:
        on_progress("Reading ORMP build…")
    exported = export_training_frame(
        build.path,
        feature_columns=feature_columns,
        label_type=label_type,  # type: ignore[arg-type]
        horizon_min=int(horizon_min),
        from_date=from_date or None,
        to_date=to_date or None,
    )
    if not exported.get("ok"):
        return {"ok": False, "error": exported.get("error") or "Export failed", **exported}

    df = exported["df"]
    label_col = exported["label_column"]
    feats = list(exported["feature_columns"])
    data_dir = chart_data_dir(chart_dir)

    if on_progress:
        on_progress(f"Writing dataset ({len(df):,} rows)…")

    metadata = {
        "dataset_name": dataset_name,
        "market": "NIFTY",
        "dataset_kind": "ORMP",
        "export_source": "ormp_dataset_builder",
        "audit_validation_required": False,
        "ormp_version": ORMP_VERSION,
        "ormp_build_id": build.build_id,
        "ormp_build_path": build.path,
        "ormp_band_size_pct": build.band_size_pct,
        "ormp_price_source": build.price_source,
        "ormp_path_mode": build.path_mode,
        "label_type": label_type,
        "horizon_min": int(horizon_min),
        "from_date": from_date,
        "to_date": to_date,
        "prediction_targets": [label_col],
        "prediction_target_columns": [label_col],
        "target_count": 1,
        "feature_columns": feats,
        "feature_count": len(feats),
        "feature_groups": [g["id"] for g in exported.get("feature_groups") or []],
        "ormp_feature_groups": exported.get("feature_groups") or [],
        "identity_columns": ["trading_day", "timestamp", "spot_open"],
        "reference_columns": ["spot_ltp"],
        "build_summary": {
            "rows_in_range": exported["rows_in_range"],
            "rows_exported": exported["rows_exported"],
            "rows_dropped_no_future": exported["rows_dropped_no_future"],
            "label_column": label_col,
            "future_column": exported.get("future_column"),
        },
        "trading_days": None,  # filled below if cheap
        "sampling": {"interval_sec": 60, "method": "ormp_1m_bars"},
    }
    # Distinct days from frame (registry expects day dicts, not bare strings)
    try:
        day_list = sorted(str(d) for d in df["trading_day"].unique().tolist())
        metadata["trading_days"] = len(day_list)
        metadata["days"] = [
            {"trading_day": d, "market": "NIFTY", "expiry": "", "source_id": d}
            for d in day_list
        ]
    except Exception:  # noqa: BLE001
        pass

    parquet_path, json_path, parquet_bytes, json_bytes = write_dataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        rows=[],
        metadata=metadata,
        existing_df=df,
    )
    return {
        "ok": True,
        "dataset_name": dataset_name,
        "parquet_path": parquet_path,
        "json_path": json_path,
        "parquet_bytes": parquet_bytes,
        "json_bytes": json_bytes,
        "label_column": label_col,
        "feature_columns": feats,
        "rows_in_range": exported["rows_in_range"],
        "rows_exported": exported["rows_exported"],
        "rows_dropped_no_future": exported["rows_dropped_no_future"],
    }


def format_size(n: int | None) -> str:
    if n is None:
        return "—"
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"
