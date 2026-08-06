"""Colorful strategy summary header for the P&L projection popup."""
from __future__ import annotations

import tkinter as tk
from typing import Any, Callable, Mapping, Sequence

_MUTED = "#94a3b8"
_RISK = "#991b1b"
_PROFIT = "#166534"
_MARGIN = "#1e40af"
_LOT_SIZE = "#1e3a8a"
_BUY = "#1d4ed8"
_SELL = "#991b1b"
_LTP = "#166534"
_LABEL = "#64748b"
_BODY = "#334155"
_ZONE = "#166534"


def estimate_profit_zone_width_pts(
    strategy_key: str | None,
    legs: Sequence[Mapping[str, Any]],
    *,
    index: str | None = None,
    wing_steps: int = 0,
    lower_be: float | None = None,
    upper_be: float | None = None,
) -> int | None:
    if lower_be is not None and upper_be is not None:
        return int(round(float(upper_be) - float(lower_be)))
    if not strategy_key or not legs:
        return None
    from api.multi_leg_margin import (
        estimate_strategy_breakevens,
        estimate_strategy_profit_zone_bounds,
    )

    bounds = estimate_strategy_profit_zone_bounds(
        strategy_key,
        legs,
        index=index,
        wing_steps=wing_steps,
    )
    if bounds is not None:
        z_lo, z_hi, _mode = bounds
        return int(round(float(z_hi) - float(z_lo)))
    lower_be, upper_be = estimate_strategy_breakevens(
        strategy_key,
        legs,
        index=index,
        wing_steps=wing_steps,
    )
    if lower_be is not None and upper_be is not None:
        return int(round(float(upper_be) - float(lower_be)))
    return None


def _leg_side(leg: Mapping[str, Any]) -> str:
    tt = str(leg.get("transaction_type") or "").upper()
    return "BUY" if tt in ("B", "BUY") else "SELL"


def _leg_short_label(leg: Mapping[str, Any]) -> str:
    strike = leg.get("strike")
    opt = str(leg.get("option_type") or "").strip().upper()
    if strike is not None and opt in ("CE", "PE"):
        try:
            return f"{int(round(float(strike)))} {opt}"
        except (TypeError, ValueError):
            pass
    sym = str(leg.get("angel_symbol") or leg.get("trading_symbol") or "?").strip()
    return sym[-12:] if len(sym) > 12 else sym


def _leg_lot_suffix(leg: Mapping[str, Any], *, lot_size: int | None) -> str:
    try:
        qty = int(leg.get("quantity") or 0)
    except (TypeError, ValueError):
        return ""
    if qty <= 0:
        return ""
    lot = max(1, int(lot_size or 1))
    lots = qty / float(lot)
    if abs(lots - round(lots)) < 1e-6:
        n = int(round(lots))
        return f"×{n} lot" if n == 1 else f"×{n} lots"
    return f"×{lots:.1f} lots"


class StrategyProjectionHeader(tk.Frame):
    """Upper summary: legs + margin / unrealized / entry / spot (no BE or ROR — those live in the table)."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        on_refresh: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master, relief=tk.GROOVE, borderwidth=1, padx=6, pady=6)
        self._on_refresh = on_refresh

        legs_hdr = tk.Frame(self)
        legs_hdr.pack(fill=tk.X)
        tk.Label(
            legs_hdr,
            text="Legs",
            font=("Arial", 9, "bold"),
            fg=_BODY,
            anchor="w",
        ).pack(side=tk.LEFT)
        tk.Button(
            legs_hdr,
            text="Copy",
            font=("Arial", 8),
            padx=6,
            pady=0,
            command=self._copy_legs_text,
        ).pack(side=tk.RIGHT)
        if on_refresh is not None:
            tk.Button(
                legs_hdr,
                text="Refresh",
                font=("Arial", 8),
                padx=6,
                pady=0,
                command=on_refresh,
            ).pack(side=tk.RIGHT, padx=(0, 4))
        self._legs_copy_text = ""
        self._legs_text = tk.Text(
            self,
            height=4,
            font=("Arial", 9),
            wrap=tk.NONE,
            relief=tk.FLAT,
            borderwidth=0,
            padx=0,
            pady=2,
            highlightthickness=0,
        )
        self._legs_text.pack(fill=tk.X)
        self._legs_text.bind("<Key>", lambda _e: "break")
        for tag, color in (
            ("label", _LABEL),
            ("buy", _BUY),
            ("sell", _SELL),
            ("entry_px", _BODY),
            ("ltp", _LTP),
            ("muted", _MUTED),
        ):
            self._legs_text.tag_configure(tag, foreground=color)

        cash = tk.Frame(self)
        cash.pack(fill=tk.X, pady=(4, 0))
        self._margin_lbl = tk.Label(cash, text="Margin: —", font=("Arial", 9, "bold"), anchor="w", fg=_MARGIN)
        self._margin_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._unreal_lbl = tk.Label(cash, text="Unrealized: —", font=("Arial", 9, "bold"), anchor="e", fg=_MUTED)
        self._unreal_lbl.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        self._zone_width_lbl = tk.Label(
            self,
            text="Profit Zone Width: —",
            font=("Arial", 9, "bold"),
            anchor="w",
            fg=_MUTED,
        )
        self._zone_width_lbl.pack(fill=tk.X, pady=(2, 0))

        meta = tk.Frame(self)
        meta.pack(fill=tk.X, pady=(2, 0))
        self._entry_lbl = tk.Label(meta, text="Entry net: —", font=("Arial", 9), anchor="w", fg=_LABEL)
        self._entry_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._spot_lbl = tk.Label(meta, text="Spot: —", font=("Arial", 9, "bold"), anchor="e", fg="#ea580c")
        self._spot_lbl.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        self._lot_lbl = tk.Label(
            self,
            text="Lot Size: —",
            font=("Arial", 9, "bold"),
            anchor="w",
            fg=_LOT_SIZE,
        )
        self._lot_lbl.pack(fill=tk.X, pady=(3, 0))

        self._mode_lbl = tk.Label(self, text="", font=("Arial", 8), anchor="w", fg=_MUTED)
        self._mode_lbl.pack(fill=tk.X, pady=(2, 0))

    def _copy_legs_text(self) -> None:
        data = str(getattr(self, "_legs_copy_text", "") or "").strip()
        if not data:
            try:
                data = self._legs_text.get("1.0", "end-1c").strip()
            except tk.TclError:
                return
        if not data:
            return
        try:
            root = self.winfo_toplevel()
            root.clipboard_clear()
            root.clipboard_append(data)
        except tk.TclError:
            pass

    def _build_legs_copy_text(
        self,
        legs: Sequence[Mapping[str, Any]],
        *,
        lot_size: int | None,
        qty_lots: int = 1,
        ltp_by_token: Mapping[str, float] | None = None,
    ) -> str:
        del ltp_by_token
        from research.strategy_math.strategy_tracker import format_legs_copy_block

        return format_legs_copy_block(
            legs, lot_size=lot_size, qty_lots=max(1, int(qty_lots or 1))
        )

    def update_summary(self, summary: Mapping[str, Any] | None) -> None:
        if not summary:
            return
        legs_detail = summary.get("legs_detail")
        if legs_detail:
            self._legs_copy_text = str(legs_detail).strip()
        else:
            legs = list(summary.get("legs") or [])
            self._legs_copy_text = self._build_legs_copy_text(
                legs,
                lot_size=summary.get("lot_size"),
                qty_lots=int(summary.get("qty_lots") or 1),
                ltp_by_token=summary.get("ltp_by_angel_token") or {},
            )
        self._render_legs(summary)
        self._render_metrics(summary)

    def _render_legs(self, summary: Mapping[str, Any]) -> None:
        legs_detail = str(summary.get("legs_detail") or "").strip()
        legs = list(summary.get("legs") or [])
        lot_size = summary.get("lot_size")
        ltp_by_token = summary.get("ltp_by_angel_token") or {}
        try:
            self._legs_text.config(state=tk.NORMAL)
            self._legs_text.delete("1.0", tk.END)
            if legs_detail and (
                "\n" in legs_detail or legs_detail[:2].strip().startswith("1.")
            ):
                line_count = legs_detail.count("\n") + 1
                try:
                    self._legs_text.config(height=min(10, max(4, line_count)))
                except tk.TclError:
                    pass
                self._legs_text.insert(tk.END, legs_detail, "entry_px")
                self._legs_text.config(state=tk.DISABLED)
                return
            try:
                self._legs_text.config(height=4)
            except tk.TclError:
                pass
            if not legs:
                self._legs_text.insert(tk.END, "Leg 1: —", "muted")
            for i, leg in enumerate(legs[:6], start=1):
                if i > 1:
                    self._legs_text.insert(tk.END, "\n")
                self._legs_text.insert(tk.END, f"Leg {i}: ", "label")
                side = _leg_side(leg)
                self._legs_text.insert(tk.END, f"{side} ", "buy" if side == "BUY" else "sell")
                self._legs_text.insert(tk.END, f"{_leg_short_label(leg)} ", "entry_px")
                try:
                    entry_px = float(leg.get("price") or 0)
                except (TypeError, ValueError):
                    entry_px = 0.0
                if entry_px > 0:
                    self._legs_text.insert(tk.END, f"@ {entry_px:.2f} ", "entry_px")
                from research.strategy_math.strategy_tracker import leg_angel_token

                atok = leg_angel_token(leg)
                live = None
                if atok and ltp_by_token:
                    try:
                        live = float(ltp_by_token.get(atok, 0) or 0)
                    except (TypeError, ValueError):
                        live = None
                if live is not None and live > 0:
                    self._legs_text.insert(tk.END, f"LTP {live:.2f} ", "ltp")
                lot_s = _leg_lot_suffix(leg, lot_size=lot_size)
                if lot_s:
                    self._legs_text.insert(tk.END, f"({lot_s})", "muted")
            self._legs_text.config(state=tk.DISABLED)
        except tk.TclError:
            pass

    def _render_metrics(self, summary: Mapping[str, Any]) -> None:
        strategy_key = str(summary.get("strategy_key") or "") or None

        margin = summary.get("margin_est")
        try:
            if margin is not None:
                self._margin_lbl.config(text=f"Margin: ₹{float(margin):,.0f}", fg=_MARGIN)
            else:
                self._margin_lbl.config(text="Margin: —", fg=_MUTED)
        except tk.TclError:
            pass

        unreal = summary.get("unrealized")
        status = str(summary.get("status") or "").upper()
        try:
            if status == "OPEN" and unreal is not None:
                u = float(unreal)
                fg = _PROFIT if u >= 0 else _RISK
                self._unreal_lbl.config(text=f"Unrealized: ₹{u:,.0f}", fg=fg)
            elif status == "COMPLETED":
                net = summary.get("net_pnl")
                if net is not None:
                    n = float(net)
                    fg = _PROFIT if n >= 0 else _RISK
                    self._unreal_lbl.config(text=f"Net P&L: ₹{n:,.0f}", fg=fg)
                else:
                    self._unreal_lbl.config(text="Net P&L: —", fg=_MUTED)
            else:
                self._unreal_lbl.config(text="Unrealized: —", fg=_MUTED)
        except tk.TclError:
            pass

        zone_width = summary.get("profit_zone_width_pts")
        try:
            if zone_width is not None:
                self._zone_width_lbl.config(
                    text=f"Profit Zone Width: {int(zone_width):,} pts",
                    fg=_ZONE,
                )
            else:
                self._zone_width_lbl.config(text="Profit Zone Width: —", fg=_MUTED)
        except tk.TclError:
            pass

        entry_net = summary.get("entry_net_per_unit")
        try:
            if entry_net is not None:
                self._entry_lbl.config(text=f"Entry net {float(entry_net):+.2f}/unit", fg=_LABEL)
            else:
                self._entry_lbl.config(text="Entry net: —", fg=_MUTED)
        except tk.TclError:
            pass

        lot_size = summary.get("lot_size")
        try:
            if lot_size is not None:
                lot = int(lot_size)
                self._lot_lbl.config(
                    text=f"Lot Size: {lot}",
                    font=("Arial", 9, "bold"),
                    fg=_LOT_SIZE,
                )
            else:
                self._lot_lbl.config(
                    text="Lot Size: —",
                    font=("Arial", 9, "bold"),
                    fg=_MUTED,
                )
        except (TypeError, ValueError, tk.TclError):
            pass

        spot = summary.get("spot")
        try:
            if spot is not None:
                self._spot_lbl.config(text=f"Spot {float(spot):,.2f}", fg="#ea580c")
            else:
                self._spot_lbl.config(text="Spot: —", fg=_MUTED)
        except tk.TclError:
            pass

        mode_parts: list[str] = []
        if summary.get("test_mode"):
            mode_parts.append("Test")
        elif summary.get("test_mode") is False:
            mode_parts.append("Live")
        if strategy_key:
            mode_parts.append(str(summary.get("strategy_label") or strategy_key.replace("_", " ").title()))
        try:
            self._mode_lbl.config(text="  ·  ".join(mode_parts) if mode_parts else "")
        except tk.TclError:
            pass


def build_projection_summary(
    *,
    legs: Sequence[Mapping[str, Any]],
    strategy_key: str | None = None,
    strategy_label: str | None = None,
    spot: float | None = None,
    lower_be: float | None = None,
    upper_be: float | None = None,
    max_profit_hint: float | None = None,
    max_risk_hint: float | None = None,
    margin_est: float | None = None,
    unrealized: float | None = None,
    lot_size: int | None = None,
    qty_lots: int = 1,
    index: str | None = None,
    wing_steps: int = 0,
    entry: Mapping[str, Any] | None = None,
    test_mode: bool | None = None,
    status: str | None = None,
    err: bool = False,
) -> dict[str, Any]:
    from research.strategy_math.strategy_tracker import entry_net_per_unit

    if entry is not None:
        if margin_est is None:
            margin_est = entry.get("margin_est")
        if test_mode is None:
            test_mode = entry.get("test_mode")
        if status is None:
            status = entry.get("status")
        if strategy_key is None:
            strategy_key = entry.get("strategy_key")
        if strategy_label is None:
            strategy_label = entry.get("strategy_label")
        if spot is None:
            spot = entry.get("spot_at_entry")
        if index is None:
            index = entry.get("index")
        if not wing_steps:
            try:
                wing_steps = int(entry.get("wing_steps") or 0)
            except (TypeError, ValueError):
                wing_steps = 0

    legs_list = [dict(leg) for leg in legs]
    entry_net = None
    if legs_list:
        try:
            entry_net = entry_net_per_unit(legs_list)
        except (TypeError, ValueError):
            pass

    try:
        from research.strategy_math.strategy_ltp_feed import get_strategy_ltp_cache

        ltp_cache = get_strategy_ltp_cache()
    except Exception:
        ltp_cache = {}

    net_pnl = entry.get("net_pnl") if entry is not None else None
    profit_zone_width_pts = estimate_profit_zone_width_pts(
        strategy_key,
        legs_list,
        index=str(index or "NIFTY").upper() if index or strategy_key else None,
        wing_steps=wing_steps,
        lower_be=lower_be,
        upper_be=upper_be,
    )

    return {
        "legs": legs_list,
        "lot_size": lot_size,
        "qty_lots": qty_lots,
        "index": str(index or "").strip().upper() or None,
        "strategy_key": strategy_key,
        "strategy_label": strategy_label,
        "spot": spot,
        "margin_est": margin_est,
        "unrealized": unrealized,
        "net_pnl": net_pnl,
        "entry_net_per_unit": entry_net,
        "test_mode": test_mode,
        "status": status,
        "ltp_by_angel_token": ltp_cache,
        "profit_zone_width_pts": profit_zone_width_pts,
    }
