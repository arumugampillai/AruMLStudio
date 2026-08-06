"""Session OHLC / prev_close from ``token_day_meta`` (exchange SNAP_QUOTE fields)."""

from __future__ import annotations

import sqlite3
from typing import Any


_SESSION_KEYS = ("open", "high", "low", "prev_close")


def _paise_to_rupees(raw: Any, *, zero_as_zero: bool = False) -> float | None:
    """``token_day_meta`` session prices are paise (INTEGER or REAL).

    When ``zero_as_zero`` is True, exchange-latched ``0`` becomes ``0.0``
    instead of missing (used for ``day_low`` / ``option_low``).
    """
    if raw is None:
        return None
    try:
        paise = int(raw)
    except (TypeError, ValueError):
        return None
    if paise < 0:
        return None
    if paise == 0:
        return 0.0 if zero_as_zero else None
    return paise / 100.0


def load_session_ohlc_by_token(
    conn: sqlite3.Connection,
    tokens: list[str],
    *,
    as_of_date: str,
) -> dict[str, dict[str, float | None]]:
    """Return ``token → {open, high, low, prev_close}`` in rupees."""
    out: dict[str, dict[str, float | None]] = {
        str(t): {k: None for k in _SESSION_KEYS} for t in tokens if str(t).strip()
    }
    if not out:
        return out
    cols = {row[1] for row in conn.execute("PRAGMA table_info(token_day_meta)")}
    needed = {"token", "as_of_date"}
    if not needed.issubset(cols):
        return out
    select = ["token"]
    for col, key in (
        ("day_open", "open"),
        ("day_high", "high"),
        ("day_low", "low"),
        ("prev_close", "prev_close"),
    ):
        if col in cols:
            select.append(col)
    if len(select) == 1:
        return out
    placeholders = ",".join("?" for _ in out)
    rows = conn.execute(
        f"""
        SELECT {", ".join(select)}
        FROM token_day_meta
        WHERE as_of_date = ?
          AND token IN ({placeholders})
        """,
        [as_of_date, *out.keys()],
    ).fetchall()
    col_idx = {name: i for i, name in enumerate(select)}
    for row in rows:
        tok = str(row[0] or "").strip()
        if tok not in out:
            continue
        sess = out[tok]
        if "day_open" in col_idx:
            sess["open"] = _paise_to_rupees(row[col_idx["day_open"]])
        if "day_high" in col_idx:
            sess["high"] = _paise_to_rupees(row[col_idx["day_high"]])
        if "day_low" in col_idx:
            # Exchange sometimes latches day_low=0; keep 0.0 (not NULL).
            sess["low"] = _paise_to_rupees(
                row[col_idx["day_low"]], zero_as_zero=True
            )
        if "prev_close" in col_idx:
            sess["prev_close"] = _paise_to_rupees(row[col_idx["prev_close"]])
    return out


def emit_session_ohlc_features(
    raw: dict[str, Any],
    *,
    spot_session: dict[str, float | None] | None,
    option_session: dict[str, float | None] | None,
) -> dict[str, Any]:
    """Broadcast spot_* and option_* session OHLC Registry Base levels.

    ``option_low`` defaults to ``0.0`` when meta low is missing (avoids NULL
    from latched day_low=0 / absent token_day_meta).
    """
    out = dict(raw)
    spot = spot_session or {}
    opt = option_session or {}
    out["spot_open"] = spot.get("open")
    out["spot_high"] = spot.get("high")
    out["spot_low"] = spot.get("low")
    out["spot_prev_close"] = spot.get("prev_close")
    out["option_open"] = opt.get("open")
    out["option_high"] = opt.get("high")
    opt_low = opt.get("low")
    out["option_low"] = 0.0 if opt_low is None else float(opt_low)
    out["option_prev_close"] = opt.get("prev_close")
    return out


__all__ = [
    "load_session_ohlc_by_token",
    "emit_session_ohlc_features",
]
