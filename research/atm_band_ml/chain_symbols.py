"""Resolve Kotak trading symbols from option-chain trees / ML entry rows."""
from __future__ import annotations

from typing import Any, Mapping


def trading_symbol_for_token(
    token: str,
    ce_tree=None,
    pe_tree=None,
) -> str:
    """Lookup ``pTrdSymbol`` (fallback ``pSymbol``) for a neo token in CE/PE trees."""
    from api.chain_quote import normalize_neo_token
    from ui.option_chain_columns import COL_SYMBOL, COL_TOKEN, COL_TSYM

    tok = normalize_neo_token(token)
    if not tok:
        return ""
    for tree in (ce_tree, pe_tree):
        if tree is None:
            continue
        for iid in tree.get_children():
            values = tree.item(iid, "values") or ()
            if len(values) <= COL_TOKEN:
                continue
            if normalize_neo_token(values[COL_TOKEN]) != tok:
                continue
            if len(values) > COL_TSYM and values[COL_TSYM]:
                return str(values[COL_TSYM]).strip()
            if len(values) > COL_SYMBOL and values[COL_SYMBOL]:
                return str(values[COL_SYMBOL]).strip()
    return ""


def ml_entry_display_symbol(
    entry: Mapping[str, Any],
    *,
    ce_tree=None,
    pe_tree=None,
) -> str:
    """Human-readable symbol for ML paper-trade rows (not raw token id)."""
    tok = str(entry.get("token") or "").strip()
    trd = str(entry.get("trading_symbol") or "").strip()
    if trd:
        return trd
    sym = str(entry.get("symbol") or "").strip()
    if sym and not (sym.isdigit() and tok and sym == tok):
        return sym
    resolved = trading_symbol_for_token(tok, ce_tree, pe_tree)
    return resolved or sym or tok or "—"
