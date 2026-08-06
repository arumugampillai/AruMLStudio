"""Prediction dataset schema helpers for Model Lab Phase 2."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from chain_replay_ml.training.prediction_packages import PROBABILITY_OUTPUT_COLUMNS

from .target_spec import inference_core_columns

LAB_PHASE_PREDICTION = 2
LAB_SCHEMA_VERSION_PREDICTION = 12

PRED_STATUS_NOT_GENERATED = "not_generated"
PRED_STATUS_BUILDING = "building"
PRED_STATUS_READY = "ready"
PRED_STATUS_ERROR = "error"
PRED_STATUS_PAUSED = "paused"

FEATURE_STORAGE_EMBEDDED = "embedded"
FEATURE_STORAGE_REFERENCED = "referenced"

# Parent / prediction-workspace dataset classification (Research Lab Prediction Dataset).
DATASET_TYPE_SEEN = "Seen"
DATASET_TYPE_UNSEEN = "Unseen"
DATASET_TYPES = frozenset({DATASET_TYPE_SEEN, DATASET_TYPE_UNSEEN})

# Per-day permanent catalog / lifecycle metadata (prediction_day_metadata)
DAY_WAITING = "waiting"
DAY_RUNNING = "running"
DAY_COMPLETED = "completed"
DAY_PARTIAL = "partial"
DAY_FAILED = "failed"
DAY_CANCELLED = "cancelled"
DAY_SKIPPED = "skipped"


def resolve_day_completion_status(
    rows_written: int, rows_expected: int | None
) -> str:
    """``DAY_COMPLETED`` only when every dataset row got a prediction row.

    A trading day build can legitimately write fewer rows than the parent
    dataset holds (dropped NaN targets, incomplete-horizon exclusion, a
    worker that only got partway through before pause/cancel, ...). Status
    must reflect that honestly — ``Complete`` must never be shown unless
    ``rows_written >= rows_expected``. When ``rows_expected`` is unknown/zero
    (no denominator to compare against) this falls back to the legacy
    "day finished processing" meaning, since partial-ness cannot be measured.
    """
    try:
        expected = int(rows_expected) if rows_expected is not None else 0
    except (TypeError, ValueError):
        expected = 0
    if expected <= 0:
        return DAY_COMPLETED
    return DAY_COMPLETED if int(rows_written or 0) >= expected else DAY_PARTIAL

# Fixed research columns (identity + prediction + trade).
# Legacy builds also store selected features as sf_*. New builds store master_row_id
# only and join Master Dataset via PredictionFeatureStore.
CORE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("lab_uuid", "TEXT NOT NULL"),
    ("prediction_id", "TEXT NOT NULL"),
    ("trading_day", "TEXT"),
    ("timestamp", "REAL"),
    ("token", "TEXT"),
    ("strike", "REAL"),
    ("option_type", "TEXT"),
    ("expiry", "TEXT"),
    ("market", "TEXT"),
    ("current_spot", "REAL"),
    ("current_ltp", "REAL"),
    ("minutes_to_expiry", "REAL"),
    ("target_column", "TEXT"),
    ("predicted_future_ltp", "REAL"),
    ("actual_future_ltp", "REAL"),
    ("expected_move", "REAL"),
    ("actual_move", "REAL"),
    ("predicted_trend", "TEXT"),
    ("actual_trend", "TEXT"),
    ("absolute_error", "REAL"),
    ("prediction_error", "REAL"),
    ("premium_error_pct", "REAL"),
    ("direction_correct", "INTEGER"),
    ("maximum_profit", "REAL"),
    ("maximum_drawdown", "REAL"),
    ("dd_before_target", "REAL"),
    ("time_to_max_profit", "REAL"),
    ("time_to_max_drawdown", "REAL"),
    ("time_to_dd_before_target", "REAL"),
    ("time_to_target", "REAL"),
    ("target_reached", "INTEGER"),
    ("target_reached_at", "REAL"),
    ("rr_1_1_hit", "INTEGER"),
    ("rr_2_3_hit", "INTEGER"),
    ("rr_1_2_hit", "INTEGER"),
    ("rr_1_3_hit", "INTEGER"),
    ("rr_1_4_hit", "INTEGER"),
    ("max_profit_at", "REAL"),
    ("max_drawdown_at", "REAL"),
    ("exit_at", "REAL"),
    ("master_row_id", "INTEGER"),
    # Triple Barrier side-scorer columns — stable schema, NULL when unselected or disabled
    ("tb_model_name", "TEXT"),
    ("tb_label_run", "TEXT"),
    ("tb_pred_probability", "REAL"),
    ("tb_pred_class", "INTEGER"),
    # Prediction-package probability ladder — stable schema, NULL when the
    # classifier member is missing (probabilities only, never decisions).
    *tuple((col, "REAL") for col in PROBABILITY_OUTPUT_COLUMNS),
    # Confidence inference columns — Market + Replay-Based (TargetSpec)
    *inference_core_columns(),
)

CORE_COLUMN_NAMES = tuple(name for name, _ in CORE_COLUMNS if name != "id")

SUMMARY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("id", "INTEGER PRIMARY KEY CHECK (id = 1)"),
    ("lab_uuid", "TEXT NOT NULL"),
    ("status", "TEXT NOT NULL"),
    ("row_count", "INTEGER NOT NULL DEFAULT 0"),
    ("trading_days", "INTEGER NOT NULL DEFAULT 0"),
    ("start_day", "TEXT"),
    ("end_day", "TEXT"),
    ("average_error", "REAL"),
    ("average_absolute_error", "REAL"),
    ("premium_error", "REAL"),
    ("direction_accuracy", "REAL"),
    ("generation_time_sec", "REAL"),
    ("dataset_hash", "TEXT"),
    ("selected_feature_count", "INTEGER"),
    ("feature_columns_json", "TEXT"),
    ("parent_model_name", "TEXT"),
    ("parent_dataset", "TEXT"),
    ("target_column", "TEXT"),
    ("created_at", "TEXT"),
    ("error_message", "TEXT"),
    ("feature_storage_mode", "TEXT"),
    ("master_dataset_id", "TEXT"),
    ("master_db_path", "TEXT"),
    ("dataset_type", "TEXT"),
)

_FEATURE_COL_RE = re.compile(r"[^A-Za-z0-9_]")


def normalize_dataset_type(value: Any) -> str:
    """Map stored/legacy values → Seen | Unseen. Missing → Seen."""
    text = str(value or "").strip()
    if not text:
        return DATASET_TYPE_SEEN
    lowered = text.lower()
    if lowered in ("unseen", "holdout", "oot", "out_of_sample", "out-of-sample"):
        return DATASET_TYPE_UNSEEN
    if lowered in ("seen", "train", "training", "in_sample", "in-sample"):
        return DATASET_TYPE_SEEN
    if text in DATASET_TYPES:
        return text
    return DATASET_TYPE_SEEN


def sanitize_feature_column(name: str) -> str:
    """Map a selected feature name to a stable SQLite column (sf_* prefix)."""
    raw = str(name or "").strip()
    safe = _FEATURE_COL_RE.sub("_", raw)
    if not safe:
        safe = "unnamed"
    if safe[0].isdigit():
        safe = f"n_{safe}"
    return f"sf_{safe}"


def model_trained_feature_names(model: Any) -> list[str]:
    """Feature order the loaded estimator expects at predict time."""
    raw = getattr(model, "feature_names_in_", None)
    if raw is not None:
        names = [str(x) for x in list(raw) if str(x).strip()]
        if names:
            return names

    get_booster = getattr(model, "get_booster", None)
    if callable(get_booster):
        try:
            booster = get_booster()
            fn = getattr(booster, "feature_names", None)
            if fn:
                names = [str(x) for x in list(fn) if str(x).strip()]
                if names:
                    return names
        except Exception:
            pass

    # LightGBM Booster
    feat_name = getattr(model, "feature_name", None)
    if callable(feat_name):
        try:
            names = [str(x) for x in list(feat_name()) if str(x).strip()]
            if names:
                return names
        except Exception:
            pass
    inner = getattr(model, "_booster", None)
    if inner is not None:
        feat_name = getattr(inner, "feature_name", None)
        if callable(feat_name):
            try:
                names = [str(x) for x in list(feat_name()) if str(x).strip()]
                if names:
                    return names
            except Exception:
                pass
    return []


def align_features_to_model(selected: list[str], model: Any) -> list[str]:
    """
    Reorder selected features to match the trained model.

    XGBoost/LightGBM reject DataFrames whose column order differs from training
    even when the feature *set* is identical.
    """
    selected_clean = [str(f).strip() for f in selected if str(f).strip()]
    if not selected_clean:
        return []

    trained = model_trained_feature_names(model)
    if not trained:
        return selected_clean

    selected_set = set(selected_clean)
    trained_set = set(trained)

    if selected_set == trained_set:
        return list(trained)

    # Prefer trained order for the intersection; append any selected-only extras
    ordered = [f for f in trained if f in selected_set]
    extras = [f for f in selected_clean if f not in trained_set]
    return ordered + extras


def feature_column_map(features: list[str]) -> dict[str, str]:
    """feature_name -> sql column; resolve collisions with numeric suffixes."""
    used: set[str] = set()
    out: dict[str, str] = {}
    for feat in features:
        base = sanitize_feature_column(feat)
        col = base
        n = 2
        while col in used or col in CORE_COLUMN_NAMES or col == "id":
            col = f"{base}_{n}"
            n += 1
        used.add(col)
        out[str(feat)] = col
    return out


def horizon_sec_from_target(target: str | None) -> float:
    """Resolve prediction evaluation window from the regression target column.

    Examples: ``future_ltp_3m`` → 180, ``future_ltp_5m`` → 300, ``future_ltp_10m`` → 600.
    Raises if the target cannot be parsed — never silently defaults to 5 minutes.
    """
    name = str(target or "").strip().lower()
    if not name:
        raise ValueError("target_column is required to resolve prediction horizon")
    m = re.match(r"future_ltp_(\d+)m$", name)
    if m:
        return float(int(m.group(1)) * 60)
    m = re.match(r"future_ltp_(\d+)s$", name)
    if m:
        return float(int(m.group(1)))
    m = re.match(r".*_(\d+)m$", name)
    if m:
        return float(int(m.group(1)) * 60)
    m = re.match(r".*_(\d+)s$", name)
    if m:
        return float(int(m.group(1)))
    raise ValueError(f"Cannot resolve prediction horizon from target: {target!r}")


def horizon_label_from_target(target: str | None) -> str:
    """Short label for UI (e.g. ``future_ltp_5m`` → ``5m``)."""
    sec = horizon_sec_from_target(target)
    if sec >= 60 and abs(sec % 60) < 1e-9:
        return f"{int(sec // 60)}m"
    return f"{int(sec)}s"


def actual_ltp_column_from_target(target: str | None) -> str:
    """Map regression target column → prediction_meta actual LTP column.

    Examples: ``future_ltp_5m`` → ``actual_5m_ltp``, ``future_ltp_3m`` → ``actual_3m_ltp``.
    No default horizon — target is required.
    """
    name = str(target or "").strip().lower()
    if not name:
        raise ValueError("target_column is required to resolve actual LTP column")
    # Prefer structured future_ltp_* names used by the master dataset.
    if name == "future_ltp_180s":
        return "actual_3m_ltp"
    m = re.match(r"future_ltp_(\d+)(m|s)$", name)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        label = f"{n}m" if unit == "m" else f"{n}s"
        return f"actual_{label}_ltp"
    label = horizon_label_from_target(name)
    return f"actual_{label}_ltp"


def has_complete_prediction_horizon(
    *,
    timestamp: float | None,
    data_end_ts: float | None,
    horizon_sec: float,
    tol_sec: float = 0.5,
) -> bool:
    """True when ``[timestamp, timestamp + horizon]`` fits in available data.

    Driven only by the configured regression target horizon and the day's last
    available tick — never by a hardcoded market-close clock.
    """
    if timestamp is None or data_end_ts is None:
        return False
    try:
        return float(timestamp) + float(horizon_sec) <= float(data_end_ts) + float(tol_sec)
    except (TypeError, ValueError):
        return False


def prediction_horizon_cutoff_ts(
    data_end_ts: float | None,
    horizon_sec: float,
) -> float | None:
    """Latest sample timestamp that still has a complete evaluation window."""
    if data_end_ts is None:
        return None
    try:
        return float(data_end_ts) - float(horizon_sec)
    except (TypeError, ValueError):
        return None


def compute_error_metrics(
    *,
    predicted: float | None,
    actual: float | None,
    entry_ltp: float | None,
) -> dict[str, Any]:
    from chain_replay_ml.prediction_meta.outcomes import compute_prediction_quality

    quality = compute_prediction_quality(
        ensemble_mean=predicted,
        entry_ltp=entry_ltp,
        actual_ltp=actual,
    )
    abs_err = None
    premium = None
    if predicted is not None and actual is not None:
        abs_err = abs(float(predicted) - float(actual))
        # Premium error % = absolute_error / abs(actual) * 100
        if abs(float(actual)) > 1e-12:
            premium = (abs_err / abs(float(actual))) * 100.0
    pred_err = quality.get("prediction_error")
    direction = quality.get("direction_correct")
    return {
        "absolute_error": round(abs_err, 6) if abs_err is not None else None,
        "prediction_error": pred_err,
        "premium_error_pct": round(premium, 6) if premium is not None else None,
        "direction_correct": int(direction) if direction is not None else None,
    }


# Reward/Risk hit thresholds: column → minimum profit/risk ratio.
# RR a:b means risk:reward, so threshold = b/a (e.g. 2:3 → 1.5).
RR_HIT_THRESHOLDS: tuple[tuple[str, float], ...] = (
    ("rr_1_1_hit", 1.0),
    ("rr_2_3_hit", 1.5),
    ("rr_1_2_hit", 2.0),
    ("rr_1_3_hit", 3.0),
    ("rr_1_4_hit", 4.0),
)
RR_HIT_COLUMNS: tuple[str, ...] = tuple(name for name, _ in RR_HIT_THRESHOLDS)


def compute_rr_hit_labels(
    *,
    target_reached: int | None,
    maximum_profit: float | None,
    maximum_drawdown: float | None,
) -> dict[str, int]:
    """
    Reward/Risk classifier labels.

    Profit = maximum_profit, Risk = maximum_drawdown.
    Requires target_reached == 1; otherwise all labels are 0.

    Thresholds (risk:reward):
      RR 1:1 → profit >= 1.0 × risk
      RR 2:3 → profit >= 1.5 × risk
      RR 1:2 → profit >= 2.0 × risk
      RR 1:3 → profit >= 3.0 × risk
      RR 1:4 → profit >= 4.0 × risk
    """
    zero = {name: 0 for name, _ in RR_HIT_THRESHOLDS}
    if target_reached != 1:
        return dict(zero)
    try:
        profit = float(maximum_profit)  # type: ignore[arg-type]
        risk = float(maximum_drawdown)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return dict(zero)
    if risk <= 0:
        return dict(zero)
    ratio = profit / risk
    return {
        name: 1 if ratio >= threshold else 0
        for name, threshold in RR_HIT_THRESHOLDS
    }


def hash_prediction_ids(prediction_ids: list[str], *, feature_count: int, row_count: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"rows={row_count}|features={feature_count}|".encode("utf-8"))
    for pid in sorted(prediction_ids):
        digest.update(pid.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def create_prediction_dataset_sql() -> str:
    cols = ",\n                ".join(f'"{name}" {typedef}' for name, typedef in CORE_COLUMNS)
    return f"""
            CREATE TABLE IF NOT EXISTS prediction_dataset (
                {cols},
                UNIQUE(prediction_id)
            );
            CREATE INDEX IF NOT EXISTS idx_pred_lab_day
                ON prediction_dataset(lab_uuid, trading_day);
            CREATE INDEX IF NOT EXISTS idx_pred_timestamp
                ON prediction_dataset(timestamp);
            CREATE INDEX IF NOT EXISTS idx_pred_master_row_id
                ON prediction_dataset(master_row_id);
            """


def create_prediction_summary_sql() -> str:
    cols = ",\n                ".join(f'"{name}" {typedef}' for name, typedef in SUMMARY_COLUMNS)
    return f"""
            CREATE TABLE IF NOT EXISTS prediction_dataset_summary (
                {cols}
            );
            """


def create_prediction_day_metadata_sql() -> str:
    """Permanent trading-day catalog + lifecycle meta for Prediction Dataset UI."""
    return """
            CREATE TABLE IF NOT EXISTS prediction_day_metadata (
                lab_uuid TEXT NOT NULL,
                trading_day TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'waiting',
                row_count INTEGER NOT NULL DEFAULT 0,
                rows_expected INTEGER,
                progress_pct REAL,
                error_message TEXT,
                started_at TEXT,
                finished_at TEXT,
                selected INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT,
                dataset_type TEXT NOT NULL DEFAULT 'Seen',
                PRIMARY KEY (lab_uuid, trading_day)
            );
            CREATE INDEX IF NOT EXISTS idx_pred_day_metadata
                ON prediction_day_metadata(lab_uuid, status);
            """


def create_prediction_build_status_sql() -> str:
    """Backward-compatible alias — prefer create_prediction_day_metadata_sql()."""
    return create_prediction_day_metadata_sql()


def create_research_dashboard_sql() -> str:
    """Precomputed Research Dashboard summary tables (Phase 3 cache)."""
    return """
            CREATE TABLE IF NOT EXISTS research_dashboard_meta (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                schema_version INTEGER NOT NULL,
                source_row_count INTEGER NOT NULL DEFAULT 0,
                source_fingerprint TEXT NOT NULL,
                computed_at TEXT NOT NULL,
                kpi_json TEXT,
                quality_json TEXT,
                risk_json TEXT,
                error_json TEXT,
                distribution_json TEXT
            );
            CREATE TABLE IF NOT EXISTS research_dashboard_premium_band (
                band TEXT PRIMARY KEY,
                sort_order INTEGER NOT NULL DEFAULT 0,
                rows INTEGER NOT NULL DEFAULT 0,
                hit_rate REAL,
                direction_accuracy REAL,
                mae REAL,
                avg_dd_before_target REAL,
                avg_time_to_target REAL,
                premium_mae REAL
            );
            CREATE TABLE IF NOT EXISTS research_dashboard_trading_day (
                trading_day TEXT PRIMARY KEY,
                rows INTEGER NOT NULL DEFAULT 0,
                hit_rate REAL,
                direction_accuracy REAL,
                mae REAL,
                avg_dd_before_target REAL,
                avg_time_to_target REAL,
                premium_mae REAL
            );
            CREATE TABLE IF NOT EXISTS research_dashboard_feature (
                feature TEXT PRIMARY KEY,
                column_name TEXT,
                n INTEGER NOT NULL DEFAULT 0,
                low_max REAL,
                high_min REAL,
                low_json TEXT,
                medium_json TEXT,
                high_json TEXT
            );
            """
