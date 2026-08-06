"""Per-token contract metadata provider used by the tick persistence layer.

Looks up trading_symbol / name / expiry / strike / option_type / lot_size /
exchange / tick_size for a given Angel token, sourced from the in-memory
instruments dataframe and INDEX_CONFIG that ``main.py`` loads at startup.

The live tick pipeline calls ``ContractMetaProvider.lookup(token)`` on the
first tick of each token per day to lazily populate the ``token_day_meta``
table. The provider stays in memory; nothing here touches SQLite.

See ``docs/chart-ticks/replay-phase-0-schema.md`` for the design rationale.
"""
from __future__ import annotations

from typing import Any, Iterable


def _to_int_or_none(raw: Any) -> int | None:
    if raw in (None, ""):
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _to_paise_or_none(rupees: Any) -> int | None:
    """Rupees float -> paise integer. Used for strike/tick_size storage."""
    if rupees in (None, ""):
        return None
    try:
        v = float(rupees)
    except (TypeError, ValueError):
        return None
    return int(round(v * 100))


def _norm_expiry(raw: Any) -> str | None:
    """Normalize expiry to ``YYYY-MM-DD`` text. Returns None if unparseable."""
    if raw in (None, ""):
        return None
    text = str(raw).strip()
    if not text or text.lower() in ("nan", "none", "nat"):
        return None

    # Try pandas-style parse first if available (catches "28-MAY-2026", "28MAY26", "2026-05-28", etc.).
    try:
        import pandas as pd  # imported lazily; this module shouldn't depend on pandas at import time
        ts = pd.to_datetime(text, errors="coerce", dayfirst=True)
        if not pd.isna(ts):
            return ts.strftime("%Y-%m-%d")
    except Exception:
        pass

    from datetime import datetime
    for fmt in ("%d%b%Y", "%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d%b%y"):
        try:
            return datetime.strptime(text.upper(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _derive_option_type(symbol: str | None) -> str | None:
    """Return 'CE' or 'PE' from the trailing suffix of an Angel options symbol."""
    if not symbol:
        return None
    s = str(symbol).strip().upper()
    if s.endswith("CE"):
        return "CE"
    if s.endswith("PE"):
        return "PE"
    return None


class ContractMetaProvider:
    """In-memory mapping of ``token`` to its static contract metadata.

    Static fields (trading_symbol, name, expiry_date, strike_price, option_type,
    lot_size, exchange, tick_size, instrument_type) are loaded at server startup
    from the existing in-memory dataframe(s). Dynamic per-day fields (day_open,
    day_high, day_low, prev_close, circuits, 52w_high/low) are NOT in this
    provider — they come from each live tick's message body.
    """

    def __init__(self) -> None:
        self._by_token: dict[str, dict[str, Any]] = {}
        self._by_name_expiry: dict[tuple[str, str], list[str]] = {}

    def _register(self, token: str, meta: dict[str, Any]) -> None:
        self._by_token[token] = meta
        name = str(meta.get("name") or "").strip().upper()
        expiry = meta.get("expiry_date")
        if name and expiry:
            self._by_name_expiry.setdefault((name, expiry), []).append(token)

    def tokens_for_name_expiry(self, name: str, expiry: str) -> list[str]:
        return list(self._by_name_expiry.get((name.strip().upper(), expiry), ()))

    def has(self, token: Any) -> bool:
        return str(token).strip() in self._by_token

    def lookup(self, token: Any) -> dict[str, Any] | None:
        """Returns the static contract metadata dict for a token, or None."""
        return self._by_token.get(str(token).strip())

    def all_tokens(self) -> Iterable[str]:
        return self._by_token.keys()

    def populate_from_options_df(
        self,
        options_df: Any,
        *,
        exchange_seg_map: dict[str, str] | None = None,
    ) -> int:
        """Bulk-load from an options DataFrame (typically ``main.all_options``).

        Expected columns: ``token``, ``symbol``, ``name``, ``expiry``,
        ``strike`` (in rupees), ``instrumenttype``, ``exch_seg``,
        ``lotsize``, ``tick_size``. Missing columns are tolerated.

        Returns the number of rows registered.
        """
        if options_df is None or getattr(options_df, "empty", True):
            return 0
        added = 0
        cols = set(options_df.columns)
        for _, row in options_df.iterrows():
            token = str(row.get("token", "")).strip()
            if not token:
                continue
            symbol = str(row.get("symbol", "")).strip() or None
            name = str(row.get("name", "")).strip() or None
            inst_type = str(row.get("instrumenttype", "")).strip() or None
            exch_seg = (
                str(row.get("exch_seg", "")).strip().upper()
                if "exch_seg" in cols else None
            )
            if exchange_seg_map and exch_seg and exch_seg in exchange_seg_map:
                exch_seg = exchange_seg_map[exch_seg]

            self._register(token, {
                "trading_symbol": symbol,
                "name": name,
                "instrument_type": inst_type or "OPTIDX",
                "expiry_date": _norm_expiry(row.get("expiry") if "expiry" in cols else None),
                "strike_price": _to_paise_or_none(row.get("strike") if "strike" in cols else None),
                "option_type": _derive_option_type(symbol),
                "lot_size": _to_int_or_none(row.get("lotsize") if "lotsize" in cols else None),
                "exchange": exch_seg,
                "tick_size": _to_paise_or_none(
                    row.get("tick_size") if "tick_size" in cols else None
                ) or _to_int_or_none(row.get("tick_size") if "tick_size" in cols else None),
            })
            added += 1
        return added

    def register_index(
        self,
        *,
        token: Any,
        display_symbol: str,
        name: str,
        exchange: str | None = None,
    ) -> None:
        """Register a non-options index token (e.g. NIFTY spot 99926000)."""
        tok = str(token).strip()
        if not tok:
            return
        self._register(tok, {
            "trading_symbol": display_symbol,
            "name": name,
            "instrument_type": "INDEX",
            "expiry_date": None,
            "strike_price": None,
            "option_type": None,
            "lot_size": None,
            "exchange": exchange,
            "tick_size": None,
        })

    def populate_from_index_config(self, index_config: dict[str, dict[str, Any]]) -> int:
        """Register every index token defined in ``main.INDEX_CONFIG``."""
        added = 0
        for key, cfg in (index_config or {}).items():
            token = str(cfg.get("index_token") or "").strip()
            if not token:
                continue
            self.register_index(
                token=token,
                display_symbol=str(cfg.get("display_symbol") or key),
                name=str(cfg.get("title") or key).upper(),
                exchange=cfg.get("exch_seg") and str(cfg.get("exch_seg")).upper(),
            )
            added += 1
        return added

    def futures_for_name(self, name: str) -> list[dict[str, Any]]:
        """Return FUTIDX contracts for ``name`` (e.g. NIFTY) as token/symbol/expiry dicts."""
        name_u = str(name or "").strip().upper()
        if not name_u:
            return []
        out: list[dict[str, Any]] = []
        for tok, meta in self._by_token.items():
            if str(meta.get("instrument_type") or "").strip().upper() != "FUTIDX":
                continue
            if str(meta.get("name") or "").strip().upper() != name_u:
                continue
            out.append({
                "token": str(tok).strip(),
                "symbol": str(meta.get("trading_symbol") or "").strip(),
                "expiry": str(meta.get("expiry_date") or "").strip(),
            })
        return out

    def size(self) -> int:
        return len(self._by_token)

    def clear(self) -> None:
        self._by_token.clear()
        self._by_name_expiry.clear()


# Module-level singleton. main.py populates it at startup;
# tick_pipeline.py reads from it for the lazy first-tick fallback.
contract_meta_provider = ContractMetaProvider()
