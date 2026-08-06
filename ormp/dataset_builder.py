"""Build standalone ORMP research dataset from 1m NIFTY candles.

Registered artifact = identity + evaluation prices + ORMP features.
Labels are created later by Model Builder from spot_ltp / future_ltp_*.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict
from typing import Any, Callable

from .config import OrmpConfig
from .data_loader import Candle, CandleLoader
from .feature_export import FEATURE_COLUMNS, IDENTITY_COLUMNS, PRICE_COLUMNS, export_features
from .market_context import MarketContextBook, null_market_context_features
from .prices import attach_forward_ltp_prices
from .profile_engine import OrmpProfile

ProgressCb = Callable[[str, dict[str, Any]], None]


def _spot_open_from_day(candles: list[Candle]) -> float:
    """09:15 open = first session candle open (loader already session-bounded)."""
    if not candles:
        raise ValueError("no candles for day")
    return float(candles[0].open)


def process_day(
    candles: list[Candle],
    *,
    band_size_pct: float,
    price_source: str,
    path_mode: str = "snapshot",
    market_context: MarketContextBook | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run ORMP for one day; return feature rows + day validation/summary."""
    spot_open = _spot_open_from_day(candles)
    profile = OrmpProfile.create(
        spot_open,
        band_size_pct,
        path_mode=path_mode,  # type: ignore[arg-type]
        candle_interval_sec=60,
    )
    rows: list[dict[str, Any]] = []
    previous_range_width = 0.0
    mc_null = null_market_context_features()

    for candle in candles:
        # Band assignment may use close/hlc3/ohlc4; evaluation LTP is always Close.
        assign_price = candle.assignment_price(price_source)  # type: ignore[arg-type]
        profile.update(assign_price, candle.timestamp, duration_sec=60.0)
        feats = export_features(profile, previous_range_width=previous_range_width)
        previous_range_width = float(feats["ormp_range_width"])
        spot_ltp = float(candle.close)
        mc_feats = dict(mc_null)
        if market_context is not None:
            mc_feats.update(
                {
                    k: v
                    for k, v in market_context.ratios_at(candle.timestamp, spot_ltp).items()
                    if k in mc_feats
                }
            )
        feats.update(mc_feats)
        rows.append(
            {
                "trading_day": candle.trading_day,
                "timestamp": candle.timestamp,
                "spot_open": spot_open,
                "spot_ltp": spot_ltp,
                **feats,
            }
        )

    attach_forward_ltp_prices(rows)

    validation = profile.validate_time_accounting()
    summary = {
        "trading_day": candles[0].trading_day,
        "candle_count": len(candles),
        "spot_open": spot_open,
        "band_size_pct": band_size_pct,
        "band_size_points": profile.band_size_points,
        "price_source": price_source,
        "path_mode": path_mode,
        "highest_band": profile.highest_band,
        "lowest_band": profile.lowest_band,
        "unique_bands": profile.unique_band_count(),
        "total_transitions": profile.total_band_transitions,
        "validation": validation,
    }
    return rows, summary


def _ensure_output_schema(conn: sqlite3.Connection) -> None:
    cols = [(name, "TEXT NOT NULL" if name == "trading_day" else "REAL NOT NULL")
            for name in IDENTITY_COLUMNS]
    # trading_day TEXT, timestamp REAL, spot_open REAL — fix types explicitly
    cols = [
        ("trading_day", "TEXT NOT NULL"),
        ("timestamp", "REAL NOT NULL"),
        ("spot_open", "REAL NOT NULL"),
    ]
    for name in PRICE_COLUMNS:
        # spot_ltp required; future LTPs nullable
        cols.append((name, "REAL NOT NULL" if name == "spot_ltp" else "REAL"))
    for name in FEATURE_COLUMNS:
        cols.append((name, "REAL"))
    col_sql = ",\n        ".join(f'"{c}" {t}' for c, t in cols)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS ormp_samples (
            {col_sql},
            PRIMARY KEY (trading_day, timestamp)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ormp_day_summary (
            trading_day TEXT PRIMARY KEY,
            candle_count INTEGER NOT NULL,
            spot_open REAL NOT NULL,
            band_size_pct REAL NOT NULL,
            band_size_points REAL NOT NULL,
            price_source TEXT NOT NULL,
            highest_band INTEGER,
            lowest_band INTEGER,
            unique_bands INTEGER,
            total_transitions INTEGER,
            total_band_time REAL,
            trading_minutes REAL,
            validation_ok INTEGER NOT NULL,
            summary_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ormp_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _write_meta(conn: sqlite3.Connection, cfg: OrmpConfig, extra: dict[str, Any]) -> None:
    payload = {
        "config": asdict(cfg),
        "identity_columns": list(IDENTITY_COLUMNS),
        "price_columns": list(PRICE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "market_context": {
            "timeframes": list(cfg.market_context_timeframes),
            "ema_periods": list(cfg.ema_periods),
        },
        "note": (
            "Stable feature dataset. Labels are created by Model Builder "
            "from spot_ltp and future_ltp_* (not stored as spot_return_*)."
        ),
        **extra,
    }
    conn.execute(
        "INSERT OR REPLACE INTO ormp_meta(key, value) VALUES (?, ?)",
        ("build", json.dumps(payload, default=str)),
    )
    conn.commit()


def build_ormp_dataset(
    cfg: OrmpConfig,
    *,
    on_progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Build full ORMP SQLite dataset under cfg.output_dir."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    tag = (
        f"bs{str(cfg.band_size_pct).replace('.', 'p')}"
        f"_{cfg.price_source}_{cfg.path_mode}"
    )
    suffix = (cfg.artifact_suffix or "").strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    out_path = os.path.join(cfg.output_dir, f"ormp_dataset_{tag}{suffix}.db")
    if os.path.isfile(out_path):
        if suffix:
            raise FileExistsError(
                f"ORMP artifact already exists (immutable): {out_path}"
            )
        os.remove(out_path)

    loader = CandleLoader(
        cfg.candle_db_path,
        token=cfg.nifty_token,
        interval_sec=cfg.interval_sec,
    )
    days = loader.list_trading_days(from_date=cfg.from_date, to_date=cfg.to_date)
    if not days:
        raise FileNotFoundError(
            f"No trading days found in {cfg.candle_db_path} "
            f"(token={cfg.nifty_token}, interval={cfg.interval_sec})"
        )

    if on_progress:
        on_progress(
            "market_context",
            {
                "timeframes": list(cfg.market_context_timeframes),
                "ema_periods": list(cfg.ema_periods),
            },
        )
    market_context = MarketContextBook.build(
        loader,
        timeframes=cfg.market_context_timeframes,
        ema_periods=cfg.ema_periods,
        from_date=cfg.from_date,
        to_date=cfg.to_date,
        base_interval_sec=cfg.interval_sec,
    )

    t0 = time.perf_counter()
    conn = sqlite3.connect(out_path)
    try:
        _ensure_output_schema(conn)
        insert_cols = list(IDENTITY_COLUMNS) + list(PRICE_COLUMNS) + list(FEATURE_COLUMNS)
        placeholders = ",".join("?" for _ in insert_cols)
        col_list = ",".join(f'"{c}"' for c in insert_cols)
        insert_sql = f"INSERT OR REPLACE INTO ormp_samples ({col_list}) VALUES ({placeholders})"

        days_ok = 0
        days_fail = 0
        rows_written = 0
        fail_examples: list[dict[str, Any]] = []

        for i, day in enumerate(days):
            candles = loader.load_day(day)
            if not candles:
                continue
            rows, summary = process_day(
                candles,
                band_size_pct=cfg.band_size_pct,
                price_source=cfg.price_source,
                path_mode=cfg.path_mode,
                market_context=market_context,
            )
            batch = [[r.get(c) for c in insert_cols] for r in rows]
            conn.executemany(insert_sql, batch)
            val = summary["validation"]
            if val["ok"]:
                days_ok += 1
            else:
                days_fail += 1
                if len(fail_examples) < 10:
                    fail_examples.append({"trading_day": day, **val})
            conn.execute(
                """
                INSERT OR REPLACE INTO ormp_day_summary(
                    trading_day, candle_count, spot_open, band_size_pct, band_size_points,
                    price_source, highest_band, lowest_band, unique_bands, total_transitions,
                    total_band_time, trading_minutes, validation_ok, summary_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    day,
                    summary["candle_count"],
                    summary["spot_open"],
                    summary["band_size_pct"],
                    summary["band_size_points"],
                    summary["price_source"],
                    summary["highest_band"],
                    summary["lowest_band"],
                    summary["unique_bands"],
                    summary["total_transitions"],
                    val.get("total_band_time"),
                    val.get("trading_minutes"),
                    1 if val["ok"] else 0,
                    json.dumps(summary, default=str),
                ),
            )
            rows_written += len(rows)
            if on_progress and (i % 25 == 0 or i == len(days) - 1):
                on_progress(
                    "day",
                    {
                        "i": i + 1,
                        "n": len(days),
                        "trading_day": day,
                        "rows_written": rows_written,
                        "days_ok": days_ok,
                        "days_fail": days_fail,
                    },
                )
            if (i + 1) % 50 == 0:
                conn.commit()

        conn.commit()
        elapsed = time.perf_counter() - t0
        result = {
            "ok": days_fail == 0,
            "output_path": out_path,
            "days_total": len(days),
            "days_ok": days_ok,
            "days_fail": days_fail,
            "rows_written": rows_written,
            "elapsed_sec": round(elapsed, 3),
            "band_size_pct": cfg.band_size_pct,
            "price_source": cfg.price_source,
            "path_mode": cfg.path_mode,
            "fail_examples": fail_examples,
        }
        _write_meta(conn, cfg, {"result": result})
        return result
    finally:
        conn.close()
