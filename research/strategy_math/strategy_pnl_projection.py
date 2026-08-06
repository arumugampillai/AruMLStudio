"""Expiry P&L projection table for the strategy panel."""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Mapping, Sequence

from research.strategy_math.strategy_projection_header import (
    StrategyProjectionHeader,
    build_projection_summary,
)
from research.strategy_math.strategy_payoff import (
    chain_projection_strikes,
    expiry_payoff_projection_table,
    spot_inside_profit_zone,
    strategy_payoff_rupees,
)
from research.strategy_math.strategy_risk_graph import _place_beside_main

_PROJECTION_WIN: "StrategyPnlProjectionWindow | None" = None
_POLL_MS = 30
_SPOT_TICK_MS = 1000
_PROJECTION_WIN_W = 400
_PROJECTION_MIN_H = 320
_PROJECTION_CHROME_H = 228
_PROJECTION_ROW_H = 22
_STRIKE_COL_W = 175
_QTY_COL_W = 28
_PNL_COL_W = 58
_GAIN_COL_W = 58
_PROJECTION_PADX = 11
_PROJECTION_Y_OFFSET_FRAC = 0.07


def _fmt_pnl(value: float) -> str:
    return f"{value:,.0f}"


def _fmt_gain_pct(
    pnl: float,
    max_profit: float,
    *,
    max_risk: float | None = None,
    breakeven_row: bool = False,
) -> str:
    """Profit rows vs max profit; loss rows vs max defined risk."""
    if breakeven_row:
        return "0.0%"
    if pnl < -0.5:
        if max_risk is not None and max_risk > 0.5:
            pct = 100.0 * float(pnl) / float(max_risk)
        elif max_profit > 0.5:
            pct = 100.0 * float(pnl) / float(max_profit)
        else:
            return "—"
    elif max_profit <= 0.5:
        return "—"
    else:
        pct = 100.0 * float(pnl) / float(max_profit)
    if abs(pct) < 0.05:
        return "0%"
    return f"{pct:.1f}%"


def _max_risk_rupees(
    rows: Sequence[tuple[float, float]],
    legs: Sequence[Mapping[str, Any]],
    *,
    strategy_key: str | None,
    index: str | None,
    wing_steps: int,
    max_risk_hint: float | None = None,
) -> float | None:
    if max_risk_hint is not None:
        try:
            hint = float(max_risk_hint)
            if hint > 0.5:
                return hint
        except (TypeError, ValueError):
            pass
    if strategy_key and legs:
        from api.multi_leg_margin import (
            estimate_strategy_risk_rupees,
            strategy_risk_is_uncapped,
        )

        if not strategy_risk_is_uncapped(strategy_key):
            risk = estimate_strategy_risk_rupees(
                strategy_key,
                legs,
                index=index,
                wing_steps=wing_steps,
            )
            if risk is not None and risk > 0.5:
                return float(risk)
    worst = min((float(p) for _, p in rows), default=0.0)
    if worst < -0.5:
        return abs(worst)
    return None


def _fmt_strike(value: float) -> str:
    return f"{int(round(float(value))):,}"


def _be_match(strike: float, be: float | None, *, grid_step: float) -> bool:
    if be is None:
        return False
    return int(round(float(strike))) == int(round(float(be)))


def _be_row_kind(
    strike: float,
    lower_be: float | None,
    upper_be: float | None,
) -> str | None:
    if _be_match(strike, lower_be, grid_step=0):
        return "lower"
    if _be_match(strike, upper_be, grid_step=0):
        return "upper"
    return None


def _be_label_prefix(strategy_key: str | None, kind: str) -> str:
    from api.multi_leg_margin import strategy_breakeven_sides

    lower_on, upper_on = strategy_breakeven_sides(strategy_key or "")
    if (lower_on or upper_on) and lower_on != upper_on:
        return "BE"
    return "LBE" if kind == "lower" else "UBE"


def _spot_distance_label(level: float, spot: float) -> str:
    """Directional distance from spot, e.g. (↓176 pts) or (↑308 pts)."""
    delta = int(round(float(level) - float(spot)))
    pts = abs(delta)
    if delta > 0:
        return f"(↑{pts} pts)"
    if delta < 0:
        return f"(↓{pts} pts)"
    return "(0 pts)"


def _center_match(strike: float, center_strike: float | None) -> bool:
    if center_strike is None:
        return False
    return int(round(float(strike))) == int(round(float(center_strike)))


def _spot_match(strike: float, spot: float) -> bool:
    return int(round(float(strike))) == int(round(float(spot)))


def _canonical_index_strike(strike: float) -> float:
    return float(int(round(float(strike))))


def _collapse_strike_alias(
    by_strike: dict[float, float],
    level: float,
) -> None:
    """Drop rows whose rounded index level matches ``level`` but float key differs."""
    target = int(round(_canonical_index_strike(level)))
    for strike in list(by_strike):
        if int(round(float(strike))) == target and strike != float(target):
            del by_strike[strike]


def _ensure_key_level_row(
    by_strike: dict[float, float],
    legs: Sequence[Mapping[str, Any]],
    level: float,
    *,
    lot_size: int | None,
) -> None:
    key = _canonical_index_strike(level)
    _collapse_strike_alias(by_strike, key)
    by_strike[key] = float(
        strategy_payoff_rupees(legs, key, lot_size=lot_size)
    )


def _canonical_spot_strike(spot: float) -> float:
    return _canonical_index_strike(spot)


def _ensure_spot_row(
    by_strike: dict[float, float],
    legs: Sequence[Mapping[str, Any]],
    spot: float,
    *,
    lot_size: int | None,
) -> None:
    """One spot row at the rounded index level (no duplicate with grid step)."""
    spot_strike = _canonical_spot_strike(spot)
    for strike in list(by_strike):
        if _spot_match(strike, spot) and strike != spot_strike:
            del by_strike[strike]
    by_strike[spot_strike] = float(
        strategy_payoff_rupees(legs, spot_strike, lot_size=lot_size)
    )


def _resolve_center_strike(
    legs: Sequence[Mapping[str, Any]],
    *,
    strategy_key: str | None,
    center_strike: float | None,
) -> float | None:
    from api.multi_leg_margin import (
        estimate_strategy_center,
        strategy_shows_center_profit_zone,
    )

    if not strategy_key or not strategy_shows_center_profit_zone(strategy_key):
        return None
    if center_strike is not None:
        return float(center_strike)
    if legs:
        center = estimate_strategy_center(strategy_key, legs)
        if center is not None:
            return float(center)
    return None


def _resolve_breakevens(
    legs: Sequence[Mapping[str, Any]],
    *,
    strategy_key: str | None,
    index: str | None,
    wing_steps: int,
    lower_be: float | None,
    upper_be: float | None,
) -> tuple[float | None, float | None]:
    if lower_be is not None or upper_be is not None:
        return lower_be, upper_be
    if not strategy_key or not legs:
        return None, None
    from api.multi_leg_margin import estimate_strategy_breakevens

    return estimate_strategy_breakevens(
        strategy_key,
        legs,
        index=index,
        wing_steps=wing_steps,
    )


def _rows_with_key_levels(
    rows: list[tuple[float, float]],
    legs: Sequence[Mapping[str, Any]],
    *,
    spot: float,
    center_strike: float | None,
    lower_be: float | None,
    upper_be: float | None,
    lot_size: int | None,
    grid_step: float,
) -> list[tuple[float, float]]:
    by_strike: dict[float, float] = {}
    for strike, pnl in rows:
        key = _canonical_index_strike(strike)
        by_strike[key] = float(pnl)

    _ensure_spot_row(by_strike, legs, spot, lot_size=lot_size)
    if center_strike is not None and not _spot_match(center_strike, spot):
        _ensure_key_level_row(
            by_strike, legs, float(center_strike), lot_size=lot_size
        )
    for be in (lower_be, upper_be):
        if be is not None:
            _ensure_key_level_row(by_strike, legs, float(be), lot_size=lot_size)
    return sorted(by_strike.items())


def _max_profit_from_rows(
    rows: Sequence[tuple[float, float]],
    center_strike: float | None,
    *,
    max_profit_hint: float | None = None,
) -> float:
    if center_strike is not None:
        for strike, pnl in rows:
            if _center_match(strike, center_strike):
                at_center = max(float(pnl), 0.0)
                if at_center > 0.5:
                    return at_center
                break
    if max_profit_hint is not None:
        try:
            hint = float(max_profit_hint)
            if hint > 0.5:
                return hint
        except (TypeError, ValueError):
            pass
    best = max((float(p) for _, p in rows), default=0.0)
    return max(best, 0.0)


def get_projection_window() -> "StrategyPnlProjectionWindow | None":
    global _PROJECTION_WIN
    if _PROJECTION_WIN is None:
        return None
    try:
        if _PROJECTION_WIN.winfo_exists():
            return _PROJECTION_WIN
    except tk.TclError:
        pass
    _PROJECTION_WIN = None
    return None


def pnl_projection_is_open() -> bool:
    return get_projection_window() is not None


def make_index_spot_provider(
    order_panel: Any | None, index: str
) -> Callable[[], float | None]:
    def provider() -> float | None:
        if order_panel is None:
            return None
        tm = getattr(order_panel, "top_menu", None)
        st = getattr(tm, "_app_state", None) if tm is not None else None
        if st is None:
            return None
        try:
            from api.market_subscribe import spot_for_atm

            return spot_for_atm(st, index)
        except Exception:
            return None

    return provider


class StrategyPnlProjectionWindow(tk.Toplevel):
    def __init__(
        self,
        master: tk.Misc,
        *,
        strategy_label: str,
        legs: Sequence[Mapping[str, Any]],
        spot: float,
        index: str | None = None,
        lot_size: int | None = None,
        half_range: float = 500,
        step: float | None = None,
        strategy_key: str | None = None,
        wing_steps: int = 0,
        lower_be: float | None = None,
        upper_be: float | None = None,
        center_strike: float | None = None,
        max_profit_hint: float | None = None,
        max_risk_hint: float | None = None,
        legs_detail: str = "",
        pnl_detail: str = "",
        summary: Mapping[str, Any] | None = None,
        on_closed: Callable[[], None] | None = None,
        on_refresh: Callable[[], None] | None = None,
        pnl_heading_total: bool = False,
        combine_mode: bool = False,
        order_panel: Any | None = None,
        expiry_raw: str | None = None,
    ) -> None:
        super().__init__(master)
        self._on_closed = on_closed
        self._on_refresh = on_refresh
        self._pnl_heading_total = bool(pnl_heading_total)
        self._combine_mode = bool(combine_mode)
        self._projection_order_panel = order_panel
        self._projection_expiry_raw = str(expiry_raw or "").strip() or None
        self._chain_strike_view_var: tk.BooleanVar | None = None
        self._anchor_win = master
        self._refresh_gen = 0
        self._poll_after_id: str | None = None
        self._spot_tick_after_id: str | None = None
        self._spot_provider: Callable[[], float | None] | None = None
        self._live_refresh_provider: Callable[[], dict[str, Any] | None] | None = None
        self._last_legs_fp: tuple[tuple[str, float], ...] | None = None
        self._last_spot: float | None = None
        self._cached_grid_rows: list[tuple[float, float]] = []
        self._cached_legs: list[dict[str, Any]] = []
        self._cached_lot_size: int | None = None
        self._cached_grid_step: float = 50.0
        self._cached_lower_be: float | None = None
        self._cached_upper_be: float | None = None
        self._cached_center_strike: float | None = None
        self._cached_max_profit_hint: float | None = None
        self._cached_max_risk_hint: float | None = None
        self._cached_index: str | None = None
        self._cached_wing_steps: int = 0
        self._cached_strategy_key: str | None = None
        self._cached_half_range: float = 500.0
        self._cached_in_profit_zone: bool = False
        if max_profit_hint is not None:
            try:
                self._cached_max_profit_hint = float(max_profit_hint)
            except (TypeError, ValueError):
                pass
        if max_risk_hint is not None:
            try:
                self._cached_max_risk_hint = float(max_risk_hint)
            except (TypeError, ValueError):
                pass
        self.minsize(_PROJECTION_WIN_W, _PROJECTION_MIN_H)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._title_lbl = tk.Label(
            self,
            text=strategy_label,
            font=("Arial", 10, "bold"),
            anchor="w",
        )
        self._title_lbl.pack(fill=tk.X, padx=_PROJECTION_PADX, pady=(6, 0))
        header_refresh = self._invoke_refresh if on_refresh is not None else None
        self._summary_header = StrategyProjectionHeader(self, on_refresh=header_refresh)
        self._summary_header.pack(fill=tk.X, padx=_PROJECTION_PADX, pady=(4, 6))
        if summary is None and (legs_detail or pnl_detail):
            summary = build_projection_summary(
                legs=legs,
                strategy_key=strategy_key,
                strategy_label=strategy_label,
                spot=spot,
                lower_be=lower_be,
                upper_be=upper_be,
                max_profit_hint=max_profit_hint,
                max_risk_hint=max_risk_hint,
                lot_size=lot_size,
                index=index,
                wing_steps=wing_steps,
            )
        if legs_detail:
            merged = dict(summary or {})
            merged["legs_detail"] = legs_detail
            summary = merged
        self.set_summary(summary)

        from config import manipulate_prefs as mp

        if self._combine_mode:
            chain_default = mp.combine_projection_chain_strikes()
        else:
            chain_default = mp.strategy_projection_chain_strikes()
        self._chain_strike_view_var = tk.BooleanVar(value=chain_default)
        chain_row = tk.Frame(self)
        chain_row.pack(fill=tk.X, padx=_PROJECTION_PADX, pady=(0, 4))
        tk.Checkbutton(
            chain_row,
            text="Chain strikes (like option chain)",
            variable=self._chain_strike_view_var,
            font=("Arial", 8),
            command=self._on_chain_strike_view_toggle,
        ).pack(side=tk.LEFT)

        frame = tk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True, padx=_PROJECTION_PADX, pady=(0, 4))
        self._table_frame = frame

        cols = (
            ("qty", "strike", "pnl", "gain")
            if self._combine_mode
            else ("strike", "pnl", "gain")
        )
        self._tree = ttk.Treeview(frame, columns=cols, show="headings", height=10)
        self._sync_table_columns()
        pnl_heading = (
            "Total P&L (₹)"
            if self._pnl_heading_total or str(strategy_label).startswith("Combine")
            else "P&L (₹)"
        )
        self._tree.heading("pnl", text=pnl_heading)
        self._tree.heading("gain", text="P&L % of Max")
        self._tree.column("strike", width=_STRIKE_COL_W, anchor=tk.E)
        self._tree.column("pnl", width=_PNL_COL_W, anchor=tk.E)
        self._tree.column("gain", width=_GAIN_COL_W, anchor=tk.E)
        self._tree.tag_configure(
            "profit", background="#ecfdf5", foreground="#166534"
        )
        self._tree.tag_configure("loss", background="#fef2f2", foreground="#991b1b")
        self._tree.tag_configure("flat", foreground="#475569")
        self._tree.tag_configure(
            "spot",
            background="#fef9c3",
            foreground="#ea580c",
            font=("Arial", 9, "bold"),
        )
        self._tree.tag_configure(
            "lower_be", background="#bbf7d0", foreground="#14532d"
        )
        self._tree.tag_configure(
            "upper_be", background="#bbf7d0", foreground="#14532d"
        )
        self._tree.tag_configure(
            "center",
            background="#ecfdf5",
            foreground="#166534",
            font=("Arial", 9, "bold"),
        )

        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=scroll.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree_scroll = scroll
        self._bind_tree_scroll(frame, scroll)
        frame.bind("<Configure>", self._on_table_frame_configure, add="+")

        btn_row = tk.Frame(self)
        btn_row.pack(pady=(4, 8))
        tk.Button(
            btn_row, text="Close", command=self._on_close, font=("Arial", 9)
        ).pack()

        self.refresh(
            strategy_label=strategy_label,
            legs=legs,
            spot=spot,
            index=index,
            lot_size=lot_size,
            half_range=half_range,
            step=step,
            strategy_key=strategy_key,
            wing_steps=wing_steps,
            lower_be=lower_be,
            upper_be=upper_be,
            center_strike=center_strike,
            max_profit_hint=max_profit_hint,
            max_risk_hint=max_risk_hint,
            reposition=False,
            background=False,
        )
        self._schedule_place_beside_main()

    def set_summary(self, summary: Mapping[str, Any] | None) -> None:
        self._summary_header.update_summary(summary)

    def set_projection_context(
        self,
        *,
        order_panel: Any | None = None,
        expiry_raw: str | None = None,
    ) -> None:
        if order_panel is not None:
            self._projection_order_panel = order_panel
        raw = str(expiry_raw or "").strip()
        if raw:
            self._projection_expiry_raw = raw

    def _chain_strike_view_active(self) -> bool:
        var = getattr(self, "_chain_strike_view_var", None)
        return bool(var is not None and bool(var.get()))

    def _sync_table_columns(self) -> None:
        chain = self._chain_strike_view_active()
        heading = "Strike" if chain else "Expiry Spot"
        try:
            self._tree.heading("strike", text=heading)
        except (tk.TclError, AttributeError):
            pass
        if not self._combine_mode:
            return
        try:
            show_qty = chain
            self._tree.heading("qty", text="" if show_qty else "")
            self._tree.column(
                "qty",
                width=_QTY_COL_W if show_qty else 0,
                anchor=tk.CENTER,
                stretch=tk.NO,
                minwidth=0,
            )
        except (tk.TclError, AttributeError):
            pass

    def _sync_strike_column_heading(self) -> None:
        self._sync_table_columns()

    def _on_chain_strike_view_toggle(self) -> None:
        var = getattr(self, "_chain_strike_view_var", None)
        if var is None:
            return
        try:
            from config import manipulate_prefs as mp

            pref_key = (
                "combine_projection_chain_strikes"
                if self._combine_mode
                else "strategy_projection_chain_strikes"
            )
            mp.save({pref_key: bool(var.get())})
        except Exception:
            pass
        self._sync_table_columns()
        self._reapply_chain_view_from_cache()

    def _combine_tail_cache_for_rows(
        self, rows: Sequence[tuple[float, float]]
    ) -> dict[str, Any]:
        if not self._combine_mode or not self._chain_strike_view_active():
            return {}
        if not self._cached_legs or not rows:
            return {}
        from research.strategy_math.strategy_uncapped_tail import compute_uncapped_tail_cache

        pnl_by_strike = {float(s): float(p) for s, p in rows}
        lot_size = self._cached_lot_size
        return compute_uncapped_tail_cache(
            self._cached_legs,
            pnl_by_strike,
            lot_size=lot_size,
        )

    def _row_qty_pnl_display(
        self,
        expiry_strike: float,
        pnl: float,
        *,
        display_pnl: float,
        be_row: bool,
        tail_cache: dict[str, Any],
    ) -> tuple[str, str]:
        from research.strategy_math.strategy_uncapped_tail import uncapped_lots_at_strike

        pnl_txt = _fmt_pnl(display_pnl)
        qty_txt = ""
        if self._combine_mode and self._chain_strike_view_active() and not be_row:
            lots = uncapped_lots_at_strike(expiry_strike, tail_cache, pnl)
            if lots:
                qty_txt = f"×{lots}"
                pnl_txt = "∞"
        return qty_txt, pnl_txt

    def _table_row_values(
        self,
        *,
        expiry_strike: float,
        spot: float,
        grid_step: float,
        lower_be: float | None,
        upper_be: float | None,
        center_strike: float | None,
        strategy_key: str | None,
        chain_view: bool,
        pnl_txt: str,
        gain_txt: str,
        qty_txt: str,
    ) -> tuple:
        strike_cell = self._strike_cell(
            expiry_strike,
            spot=spot,
            grid_step=grid_step,
            lower_be=lower_be,
            upper_be=upper_be,
            center_strike=center_strike,
            strategy_key=strategy_key,
            chain_view=chain_view,
        )
        if self._combine_mode:
            return (qty_txt, strike_cell, pnl_txt, gain_txt)
        return (strike_cell, pnl_txt, gain_txt)

    def _resolve_chain_available_strikes(self, index: str | None) -> list[float]:
        expiry_raw = getattr(self, "_projection_expiry_raw", None)
        op = getattr(self, "_projection_order_panel", None)
        if not expiry_raw or op is None or not index:
            return []
        tm = getattr(op, "top_menu", None)
        st = getattr(tm, "_app_state", None) if tm is not None else None
        if st is None:
            return []
        options_df = getattr(st, "options_df", None)
        if options_df is None or getattr(options_df, "empty", True):
            return []
        try:
            from research.strategy_math.combine_spread_scanner import chain_strikes_for_expiry

            return chain_strikes_for_expiry(
                options_df,
                index=str(index),
                expiry_raw=str(expiry_raw),
            )
        except Exception:
            return []

    def _compute_projection_grid_rows(
        self,
        legs: Sequence[Mapping[str, Any]],
        spot_f: float,
        *,
        index: str | None,
        half_range: float,
        grid_step: float,
        lot_size: int | None,
        lower_be: float | None,
        upper_be: float | None,
        center_strike: float | None,
        strategy_key: str | None,
        in_profit_zone: bool,
    ) -> list[tuple[float, float]]:
        if self._chain_strike_view_active():
            strikes = chain_projection_strikes(
                spot_f,
                index=index,
                lower_be=lower_be,
                upper_be=upper_be,
                available_strikes=self._resolve_chain_available_strikes(index),
            )
            return [
                (
                    float(s),
                    float(strategy_payoff_rupees(legs, s, lot_size=lot_size)),
                )
                for s in strikes
            ]
        return list(
            expiry_payoff_projection_table(
                legs,
                spot=float(spot_f),
                index=index,
                half_range=half_range,
                step=grid_step,
                lot_size=lot_size,
                lower_be=lower_be,
                upper_be=upper_be,
                center_strike=center_strike,
                adaptive_in_profit_zone=in_profit_zone,
            )
        )

    def _reapply_chain_view_from_cache(self) -> None:
        if not self._cached_legs or self._last_spot is None:
            return
        in_profit_zone = spot_inside_profit_zone(
            float(self._last_spot),
            self._cached_lower_be,
            self._cached_upper_be,
            strategy_key=self._cached_strategy_key,
        )
        grid_rows = self._compute_projection_grid_rows(
            self._cached_legs,
            float(self._last_spot),
            index=self._cached_index,
            half_range=self._cached_half_range,
            grid_step=self._cached_grid_step,
            lot_size=self._cached_lot_size,
            lower_be=self._cached_lower_be,
            upper_be=self._cached_upper_be,
            center_strike=self._cached_center_strike,
            strategy_key=self._cached_strategy_key,
            in_profit_zone=in_profit_zone,
        )
        self._cached_grid_rows = list(grid_rows)
        self._cached_in_profit_zone = in_profit_zone
        display_rows = self._display_rows_for(
            grid_rows,
            self._cached_legs,
            float(self._last_spot),
            grid_step=self._cached_grid_step,
            lower_be=self._cached_lower_be,
            upper_be=self._cached_upper_be,
            center_strike=self._cached_center_strike,
            lot_size=self._cached_lot_size,
        )
        self._apply_table(
            self._refresh_gen,
            display_rows,
            float(self._last_spot),
            grid_step=self._cached_grid_step,
            lower_be=self._cached_lower_be,
            upper_be=self._cached_upper_be,
            center_strike=self._cached_center_strike,
            reposition=False,
        )

    def set_details(self, legs_detail: str = "", pnl_detail: str = "") -> None:
        if legs_detail:
            self._summary_header._legs_copy_text = str(legs_detail).strip()

    def _schedule_place_beside_main(self) -> None:
        try:
            self.after(80, self._place_beside_main)
        except tk.TclError:
            self._place_beside_main()

    def _tree_rows_for_height(self, window_height: int) -> int:
        rows = (int(window_height) - _PROJECTION_CHROME_H) // _PROJECTION_ROW_H
        return max(6, rows)

    def _sync_tree_height_to_viewport(self) -> None:
        """Size visible tree rows to the table area (header may vary, e.g. combine)."""
        try:
            self.update_idletasks()
            fh = int(self._table_frame.winfo_height())
        except (tk.TclError, TypeError, ValueError, AttributeError):
            return
        if fh < _PROJECTION_ROW_H * 4:
            try:
                fh = max(0, int(self.winfo_height()) - _PROJECTION_CHROME_H)
            except (tk.TclError, TypeError, ValueError):
                return
        rows = max(6, min(48, fh // _PROJECTION_ROW_H))
        try:
            self._tree.configure(height=rows)
        except tk.TclError:
            pass

    def _on_table_frame_configure(self, _event: tk.Event | None = None) -> None:
        self._sync_tree_height_to_viewport()

    def _bind_tree_scroll(self, frame: tk.Frame, scroll: ttk.Scrollbar) -> None:
        def _scroll_units(delta: int) -> None:
            try:
                self._tree.yview_scroll(delta, "units")
            except tk.TclError:
                pass

        def _on_mousewheel(event: tk.Event) -> str:
            if getattr(event, "delta", 0):
                _scroll_units(int(-1 * (event.delta / 120)))
            return "break"

        def _on_mousewheel_up(_event: tk.Event) -> str:
            _scroll_units(-1)
            return "break"

        def _on_mousewheel_down(_event: tk.Event) -> str:
            _scroll_units(1)
            return "break"

        for widget in (self._tree, frame, scroll):
            widget.bind("<MouseWheel>", _on_mousewheel, add="+")
            widget.bind("<Button-4>", _on_mousewheel_up, add="+")
            widget.bind("<Button-5>", _on_mousewheel_down, add="+")

    def _sync_height_to_main(self) -> int | None:
        """Match projection window height to the main app; size table rows to fit."""
        try:
            self._anchor_win.update_idletasks()
            h_main = int(self._anchor_win.winfo_height())
        except (tk.TclError, TypeError, ValueError):
            return None
        if h_main < _PROJECTION_MIN_H:
            h_main = _PROJECTION_MIN_H
        try:
            self.geometry(f"{_PROJECTION_WIN_W}x{h_main}")
            self.update_idletasks()
            self._sync_tree_height_to_viewport()
            self.update_idletasks()
        except tk.TclError:
            return None
        return h_main

    def _projection_y_offset(self) -> int:
        """Shift projection top upward by a fraction of main window height."""
        try:
            self._anchor_win.update_idletasks()
            h_main = int(self._anchor_win.winfo_height())
            return -int(h_main * _PROJECTION_Y_OFFSET_FRAC)
        except (tk.TclError, TypeError, ValueError):
            return 0

    def _place_projection_beside_main(self) -> None:
        _place_beside_main(
            self._anchor_win, self, y_offset=self._projection_y_offset()
        )

    def _bind_main_height_sync(self) -> None:
        if getattr(self, "_height_sync_bound", False):
            return
        self._height_sync_bound = True
        last_h: list[int] = [0]

        def _on_main_configure(event: tk.Event) -> None:
            if event.widget is not self._anchor_win:
                return
            try:
                if not self.winfo_exists():
                    return
            except tk.TclError:
                return
            try:
                h = int(event.height)
            except (TypeError, ValueError):
                return
            if abs(h - last_h[0]) < 6:
                return
            last_h[0] = h
            self._sync_height_to_main()
            self._place_projection_beside_main()

        try:
            self._anchor_win.bind("<Configure>", _on_main_configure, add="+")
        except tk.TclError:
            pass

    def _place_beside_main(self) -> None:
        self._sync_height_to_main()
        self._place_projection_beside_main()
        self._bind_main_height_sync()

    def _cancel_spot_tick(self) -> None:
        if self._spot_tick_after_id is not None:
            try:
                self.after_cancel(self._spot_tick_after_id)
            except tk.TclError:
                pass
            self._spot_tick_after_id = None

    def _schedule_live_tick(self) -> None:
        self._cancel_spot_tick()
        if self._live_refresh_provider is None and self._spot_provider is None:
            return
        try:
            self._spot_tick_after_id = self.after(_SPOT_TICK_MS, self._on_live_tick)
        except tk.TclError:
            pass

    def set_live_refresh_provider(
        self, provider: Callable[[], dict[str, Any] | None]
    ) -> None:
        self._live_refresh_provider = provider
        try:
            payload = provider()
            if payload:
                self._apply_live_refresh(payload)
        except Exception:
            pass
        self._schedule_live_tick()

    def set_spot_provider(self, provider: Callable[[], float | None]) -> None:
        self._spot_provider = provider
        if self._live_refresh_provider is not None:
            return
        try:
            raw = provider()
            if raw is not None:
                self.refresh_spot_only(float(raw))
        except Exception:
            pass
        self._schedule_live_tick()

    def _legs_fingerprint(
        self, legs: Sequence[Mapping[str, Any]]
    ) -> tuple[tuple[str, float], ...]:
        fp: list[tuple[str, float]] = []
        for leg in legs:
            tok = str(leg.get("token") or leg.get("instrument_token") or "")
            try:
                px = round(float(leg.get("price") or 0), 2)
            except (TypeError, ValueError):
                px = 0.0
            fp.append((tok, px))
        return tuple(fp)

    def _apply_live_refresh(self, payload: Mapping[str, Any]) -> None:
        legs = list(payload.get("legs") or [])
        if not legs:
            return
        legs_detail = str(payload.get("legs_detail") or "")
        pnl_detail = str(payload.get("pnl_detail") or "")
        summary = payload.get("summary")
        if summary:
            self.set_summary(summary)
        elif legs_detail or pnl_detail:
            self.set_details(legs_detail, pnl_detail)

        fp = self._legs_fingerprint(legs)
        spot_raw = payload.get("spot")
        if spot_raw is None:
            spot_f = self._last_spot if self._last_spot is not None else 0.0
        else:
            spot_f = self._live_spot(float(spot_raw))

        prices_changed = fp != self._last_legs_fp
        self._last_legs_fp = fp

        if prices_changed or not self._cached_grid_rows:
            self.refresh(
                strategy_label=str(
                    payload.get("strategy_label") or self._title_lbl.cget("text")
                ),
                legs=legs,
                spot=float(spot_f),
                index=payload.get("index"),
                lot_size=payload.get("lot_size"),
                half_range=float(payload.get("half_range", 500)),
                step=payload.get("step"),
                strategy_key=payload.get("strategy_key"),
                wing_steps=int(payload.get("wing_steps") or 0),
                lower_be=payload.get("lower_be"),
                upper_be=payload.get("upper_be"),
                center_strike=payload.get("center_strike"),
                max_profit_hint=payload.get("max_profit_hint", payload.get("max_profit")),
                max_risk_hint=payload.get("max_risk_hint", payload.get("max_risk")),
                reposition=False,
                background=True,
            )
        else:
            self.refresh_spot_only(float(spot_f))

    def _on_live_tick(self) -> None:
        self._spot_tick_after_id = None
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        if self._live_refresh_provider is not None:
            try:
                payload = self._live_refresh_provider()
                if payload:
                    self._apply_live_refresh(payload)
            except Exception:
                pass
        elif self._spot_provider is not None:
            try:
                raw = self._spot_provider()
                if raw is not None:
                    self.refresh_spot_only(float(raw))
            except Exception:
                pass
        self._schedule_live_tick()

    def _schedule_spot_tick(self) -> None:
        self._schedule_live_tick()

    def _on_spot_tick(self) -> None:
        self._on_live_tick()

    def _live_spot(self, fallback: float) -> float:
        if self._spot_provider is not None:
            try:
                raw = self._spot_provider()
                if raw is not None:
                    return float(raw)
            except Exception:
                pass
        return float(fallback)

    def _display_rows_for(
        self,
        grid_rows: Sequence[tuple[float, float]],
        legs: Sequence[Mapping[str, Any]],
        spot: float,
        *,
        grid_step: float,
        lower_be: float | None,
        upper_be: float | None,
        center_strike: float | None,
        lot_size: int | None,
    ) -> list[tuple[float, float]]:
        return _rows_with_key_levels(
            list(grid_rows),
            legs,
            spot=float(spot),
            center_strike=center_strike,
            lower_be=lower_be,
            upper_be=upper_be,
            lot_size=lot_size,
            grid_step=grid_step,
        )

    def _store_table_cache(
        self,
        grid_rows: Sequence[tuple[float, float]],
        legs: Sequence[Mapping[str, Any]],
        spot: float,
        *,
        grid_step: float,
        lower_be: float | None,
        upper_be: float | None,
        center_strike: float | None,
        lot_size: int | None,
        half_range: float = 500.0,
        in_profit_zone: bool = False,
    ) -> None:
        self._cached_grid_rows = [(float(s), float(p)) for s, p in grid_rows]
        self._cached_legs = [dict(leg) for leg in legs]
        self._cached_lot_size = lot_size
        self._cached_grid_step = float(grid_step)
        self._cached_lower_be = lower_be
        self._cached_upper_be = upper_be
        self._cached_center_strike = center_strike
        self._cached_half_range = float(half_range)
        self._cached_in_profit_zone = bool(in_profit_zone)
        self._last_spot = float(spot)

    def _grid_rows_for_spot(
        self,
        spot: float,
        legs: Sequence[Mapping[str, Any]],
        *,
        index: str | None,
        half_range: float,
        grid_step: float,
        lot_size: int | None,
        lower_be: float | None,
        upper_be: float | None,
        center_strike: float | None,
        strategy_key: str | None = None,
    ) -> tuple[list[tuple[float, float]], bool]:
        in_profit_zone = spot_inside_profit_zone(
            spot, lower_be, upper_be, strategy_key=strategy_key
        )
        rows = self._compute_projection_grid_rows(
            legs,
            float(spot),
            index=index,
            half_range=half_range,
            grid_step=grid_step,
            lot_size=lot_size,
            lower_be=lower_be,
            upper_be=upper_be,
            center_strike=center_strike,
            strategy_key=strategy_key,
            in_profit_zone=in_profit_zone,
        )
        return list(rows), in_profit_zone

    def refresh_spot_only(self, spot: float) -> None:
        """Move spot row; rebuild grid if spot crosses the profit-zone boundary."""
        if not self._cached_grid_rows or not self._cached_legs:
            return
        spot_f = self._live_spot(float(spot))
        in_profit_zone = spot_inside_profit_zone(
            spot_f,
            self._cached_lower_be,
            self._cached_upper_be,
            strategy_key=self._cached_strategy_key,
        )
        if in_profit_zone != self._cached_in_profit_zone:
            grid_rows, in_profit_zone = self._grid_rows_for_spot(
                spot_f,
                self._cached_legs,
                index=self._cached_index,
                half_range=self._cached_half_range,
                grid_step=self._cached_grid_step,
                lot_size=self._cached_lot_size,
                lower_be=self._cached_lower_be,
                upper_be=self._cached_upper_be,
                center_strike=self._cached_center_strike,
                strategy_key=self._cached_strategy_key,
            )
            self._cached_in_profit_zone = in_profit_zone
            self._cached_grid_rows = grid_rows
        rows = self._display_rows_for(
            self._cached_grid_rows,
            self._cached_legs,
            spot_f,
            grid_step=self._cached_grid_step,
            lower_be=self._cached_lower_be,
            upper_be=self._cached_upper_be,
            center_strike=self._cached_center_strike,
            lot_size=self._cached_lot_size,
        )
        self._last_spot = spot_f
        self._apply_table(
            self._refresh_gen,
            rows,
            spot_f,
            grid_step=self._cached_grid_step,
            lower_be=self._cached_lower_be,
            upper_be=self._cached_upper_be,
            center_strike=self._cached_center_strike,
            reposition=False,
        )

    def _cancel_poll(self) -> None:
        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            self._poll_after_id = None

    def _invoke_refresh(self) -> None:
        cb = self._on_refresh
        if cb is not None:
            cb()

    def _on_close(self) -> None:
        self._refresh_gen += 1
        self._cancel_poll()
        self._cancel_spot_tick()
        cb = self._on_closed
        self._on_closed = None
        try:
            self.destroy()
        except tk.TclError:
            pass
        if cb is not None:
            try:
                cb()
            except Exception:
                pass

    def _row_tags(
        self,
        expiry_strike: float,
        pnl: float,
        spot: float,
        *,
        grid_step: float,
        lower_be: float | None,
        upper_be: float | None,
        center_strike: float | None,
    ) -> tuple[str, ...]:
        if _be_match(expiry_strike, lower_be, grid_step=grid_step):
            return ("lower_be",)
        if _be_match(expiry_strike, upper_be, grid_step=grid_step):
            return ("upper_be",)
        if _spot_match(expiry_strike, spot):
            return ("spot",)
        if _center_match(expiry_strike, center_strike):
            return ("center",)
        if pnl > 0.5:
            pnl_tag = "profit"
        elif pnl < -0.5:
            pnl_tag = "loss"
        else:
            pnl_tag = "flat"
        return (pnl_tag,)

    def _strike_cell(
        self,
        expiry_strike: float,
        *,
        spot: float,
        grid_step: float,
        lower_be: float | None,
        upper_be: float | None,
        center_strike: float | None,
        strategy_key: str | None = None,
        chain_view: bool = False,
    ) -> str:
        strike_s = _fmt_strike(expiry_strike)
        dist = _spot_distance_label(expiry_strike, spot)
        be_kind = _be_row_kind(expiry_strike, lower_be, upper_be)
        if chain_view:
            if be_kind == "lower":
                prefix = _be_label_prefix(strategy_key, "lower")
                return f"{prefix} {strike_s} {dist}"
            if be_kind == "upper":
                prefix = _be_label_prefix(strategy_key, "upper")
                return f"{prefix} {strike_s} {dist}"
            if _spot_match(expiry_strike, spot):
                return f"▶ {strike_s} (Spot)"
            return strike_s
        if be_kind == "lower":
            prefix = _be_label_prefix(strategy_key, "lower")
            return f"{prefix} {strike_s} {dist}"
        if be_kind == "upper":
            prefix = _be_label_prefix(strategy_key, "upper")
            return f"{prefix} {strike_s} {dist}"
        if _spot_match(expiry_strike, spot):
            return f"▶ {strike_s} (Spot)"
        if _center_match(expiry_strike, center_strike):
            return f"Center {strike_s} {dist}"
        return strike_s

    def _apply_table(
        self,
        gen: int,
        rows: Sequence[tuple[float, float]],
        spot: float,
        *,
        grid_step: float,
        lower_be: float | None,
        upper_be: float | None,
        center_strike: float | None,
        reposition: bool,
    ) -> None:
        if gen != self._refresh_gen:
            return
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        children = self._tree.get_children()
        if children:
            self._tree.delete(*children)
        max_profit = _max_profit_from_rows(
            rows,
            center_strike,
            max_profit_hint=self._cached_max_profit_hint,
        )
        max_risk = _max_risk_rupees(
            rows,
            self._cached_legs,
            strategy_key=self._cached_strategy_key,
            index=self._cached_index,
            wing_steps=self._cached_wing_steps,
            max_risk_hint=self._cached_max_risk_hint,
        )
        spot_iid: str | None = None
        strategy_key = self._cached_strategy_key
        chain_view = self._chain_strike_view_active()
        tail_cache = self._combine_tail_cache_for_rows(rows)
        for expiry_strike, pnl in rows:
            be_row = _be_row_kind(expiry_strike, lower_be, upper_be) is not None
            display_pnl = 0.0 if be_row else pnl
            qty_txt, pnl_txt = self._row_qty_pnl_display(
                expiry_strike,
                pnl,
                display_pnl=display_pnl,
                be_row=be_row,
                tail_cache=tail_cache,
            )
            tags = self._row_tags(
                expiry_strike,
                pnl,
                spot,
                grid_step=grid_step,
                lower_be=lower_be,
                upper_be=upper_be,
                center_strike=center_strike,
            )
            iid = self._tree.insert(
                "",
                tk.END,
                values=self._table_row_values(
                    expiry_strike=expiry_strike,
                    spot=spot,
                    grid_step=grid_step,
                    lower_be=lower_be,
                    upper_be=upper_be,
                    center_strike=center_strike,
                    strategy_key=strategy_key,
                    chain_view=chain_view,
                    pnl_txt=pnl_txt,
                    gain_txt=_fmt_gain_pct(
                        display_pnl,
                        max_profit,
                        max_risk=max_risk,
                        breakeven_row=be_row,
                    ),
                    qty_txt=qty_txt,
                ),
                tags=tags,
            )
            if "spot" in tags:
                spot_iid = iid
        if not rows:
            empty = ("—", "—", "—", "—") if self._combine_mode else ("—", "—", "—")
            self._tree.insert("", tk.END, values=empty)
        elif spot_iid is not None:
            try:
                self._tree.see(spot_iid)
            except tk.TclError:
                pass
        try:
            self._sync_tree_height_to_viewport()
        except Exception:
            pass
        if reposition:
            self._place_beside_main()

    def _finish_refresh(
        self,
        gen: int,
        grid_rows: Sequence[tuple[float, float]],
        legs_snapshot: Sequence[Mapping[str, Any]],
        spot_f: float,
        *,
        grid_step: float,
        lower_be: float | None,
        upper_be: float | None,
        center_strike: float | None,
        lot_size: int | None,
        half_range: float = 500.0,
        reposition: bool,
    ) -> None:
        live_spot = self._live_spot(spot_f)
        in_profit_zone = spot_inside_profit_zone(
            live_spot, lower_be, upper_be, strategy_key=self._cached_strategy_key
        )
        display_rows = self._display_rows_for(
            grid_rows,
            legs_snapshot,
            live_spot,
            grid_step=grid_step,
            lower_be=lower_be,
            upper_be=upper_be,
            center_strike=center_strike,
            lot_size=lot_size,
        )
        self._store_table_cache(
            grid_rows,
            legs_snapshot,
            live_spot,
            grid_step=grid_step,
            lower_be=lower_be,
            upper_be=upper_be,
            center_strike=center_strike,
            lot_size=lot_size,
            half_range=half_range,
            in_profit_zone=in_profit_zone,
        )
        self._apply_table(
            gen,
            display_rows,
            live_spot,
            grid_step=grid_step,
            lower_be=lower_be,
            upper_be=upper_be,
            center_strike=center_strike,
            reposition=reposition,
        )

    def _poll_refresh_result(
        self,
        result_q: queue.Queue,
        gen: int,
        spot_f: float,
        grid_step: float,
        lower_be: float | None,
        upper_be: float | None,
        center_strike: float | None,
        *,
        half_range: float = 500.0,
        reposition: bool,
    ) -> None:
        self._poll_after_id = None
        if gen != self._refresh_gen:
            return
        try:
            done_gen, grid_rows, _rows = result_q.get_nowait()
        except queue.Empty:
            try:
                self._poll_after_id = self.after(
                    _POLL_MS,
                    lambda: self._poll_refresh_result(
                        result_q,
                        gen,
                        spot_f,
                        grid_step,
                        lower_be,
                        upper_be,
                        center_strike,
                        half_range=half_range,
                        reposition=reposition,
                    ),
                )
            except tk.TclError:
                pass
            return
        if done_gen != self._refresh_gen:
            return
        if not grid_rows:
            return
        self._finish_refresh(
            gen,
            grid_rows,
            self._cached_legs or [],
            spot_f,
            grid_step=grid_step,
            lower_be=lower_be,
            upper_be=upper_be,
            center_strike=center_strike,
            lot_size=self._cached_lot_size,
            half_range=half_range,
            reposition=reposition,
        )

    def refresh(
        self,
        *,
        strategy_label: str,
        legs: Sequence[Mapping[str, Any]],
        spot: float,
        index: str | None = None,
        lot_size: int | None = None,
        half_range: float = 500,
        step: float | None = None,
        strategy_key: str | None = None,
        wing_steps: int = 0,
        lower_be: float | None = None,
        upper_be: float | None = None,
        center_strike: float | None = None,
        max_profit_hint: float | None = None,
        max_risk_hint: float | None = None,
        reposition: bool = True,
        background: bool = True,
    ) -> None:
        from api.multi_leg_margin import strike_step_for_index

        self._refresh_gen += 1
        gen = self._refresh_gen
        self._cancel_poll()

        self.title(f"P&L Projection — {strategy_label}")
        self._title_lbl.config(text=strategy_label)

        grid_step = float(step) if step is not None else float(
            strike_step_for_index(index)
        )
        spot_f = self._live_spot(float(spot))
        legs_snapshot = [dict(leg) for leg in legs]
        lower_be, upper_be = _resolve_breakevens(
            legs_snapshot,
            strategy_key=strategy_key,
            index=index,
            wing_steps=wing_steps,
            lower_be=lower_be,
            upper_be=upper_be,
        )
        if center_strike is None:
            center_strike = _resolve_center_strike(
                legs_snapshot,
                strategy_key=strategy_key,
                center_strike=None,
            )
        else:
            center_strike = _resolve_center_strike(
                legs_snapshot,
                strategy_key=strategy_key,
                center_strike=center_strike,
            )
        self._cached_legs = legs_snapshot
        self._cached_lot_size = lot_size
        self._cached_strategy_key = strategy_key
        self._cached_index = str(index).upper() if index else None
        self._cached_wing_steps = int(wing_steps or 0)
        if max_profit_hint is not None:
            try:
                self._cached_max_profit_hint = float(max_profit_hint)
            except (TypeError, ValueError):
                self._cached_max_profit_hint = None
        if max_risk_hint is not None:
            try:
                self._cached_max_risk_hint = float(max_risk_hint)
            except (TypeError, ValueError):
                self._cached_max_risk_hint = None

        def compute() -> list[tuple[float, float]]:
            in_profit_zone = spot_inside_profit_zone(
                spot_f, lower_be, upper_be, strategy_key=strategy_key
            )
            return self._compute_projection_grid_rows(
                legs_snapshot,
                spot_f,
                index=index,
                half_range=half_range,
                grid_step=grid_step,
                lot_size=lot_size,
                lower_be=lower_be,
                upper_be=upper_be,
                center_strike=center_strike,
                strategy_key=strategy_key,
                in_profit_zone=in_profit_zone,
            )

        if background:

            def worker(result_q: queue.Queue, worker_gen: int) -> None:
                try:
                    grid_rows = compute()
                    result_q.put((worker_gen, grid_rows, []))
                except Exception:
                    pass

            result_q: queue.Queue = queue.Queue(maxsize=1)
            threading.Thread(
                target=worker, args=(result_q, gen), daemon=True
            ).start()
            try:
                self._poll_after_id = self.after(
                    _POLL_MS,
                    lambda: self._poll_refresh_result(
                        result_q,
                        gen,
                        spot_f,
                        grid_step,
                        lower_be,
                        upper_be,
                        center_strike,
                        half_range=half_range,
                        reposition=reposition,
                    ),
                )
            except tk.TclError:
                pass
            return

        grid_rows = compute()
        self._finish_refresh(
            gen,
            grid_rows,
            legs_snapshot,
            spot_f,
            grid_step=grid_step,
            lower_be=lower_be,
            upper_be=upper_be,
            center_strike=center_strike,
            lot_size=lot_size,
            half_range=half_range,
            reposition=reposition,
        )
        try:
            self.lift()
        except tk.TclError:
            pass


def show_strategy_pnl_projection(
    master: tk.Misc,
    *,
    strategy_label: str,
    legs: Sequence[Mapping[str, Any]],
    spot: float | None = None,
    index: str | None = None,
    lot_size: int | None = None,
    half_range: float = 500,
    step: float | None = None,
    strategy_key: str | None = None,
    wing_steps: int = 0,
    lower_be: float | None = None,
    upper_be: float | None = None,
    center_strike: float | None = None,
    max_profit_hint: float | None = None,
    max_risk_hint: float | None = None,
    spot_provider: Callable[[], float | None] | None = None,
    live_refresh_provider: Callable[[], dict[str, Any] | None] | None = None,
    legs_detail: str = "",
    pnl_detail: str = "",
    summary: Mapping[str, Any] | None = None,
    order_panel: Any | None = None,
    expiry_raw: str | None = None,
) -> None:
    global _PROJECTION_WIN
    if spot is None:
        for leg in legs:
            try:
                spot = float(leg.get("strike"))
                break
            except (TypeError, ValueError):
                continue
    if spot is None:
        raise ValueError("spot unavailable")

    def _release_strategy_win() -> None:
        global _PROJECTION_WIN
        _PROJECTION_WIN = None

    try:
        if _PROJECTION_WIN is not None and _PROJECTION_WIN.winfo_exists():
            _PROJECTION_WIN.destroy()
    except tk.TclError:
        _release_strategy_win()
    _PROJECTION_WIN = StrategyPnlProjectionWindow(
        master,
        strategy_label=strategy_label,
        legs=legs,
        spot=float(spot),
        index=index,
        lot_size=lot_size,
        half_range=half_range,
        step=step,
        strategy_key=strategy_key,
        wing_steps=wing_steps,
        lower_be=lower_be,
        upper_be=upper_be,
        center_strike=center_strike,
        max_profit_hint=max_profit_hint,
        max_risk_hint=max_risk_hint,
        legs_detail=legs_detail,
        pnl_detail=pnl_detail,
        summary=summary,
        on_closed=_release_strategy_win,
        order_panel=order_panel,
        expiry_raw=expiry_raw,
    )
    if live_refresh_provider is not None:
        _PROJECTION_WIN.set_live_refresh_provider(live_refresh_provider)
    elif spot_provider is not None:
        _PROJECTION_WIN.set_spot_provider(spot_provider)


def make_entry_live_refresh_provider(
    entry: Mapping[str, Any],
    *,
    order_panel: Any | None = None,
    ltp_by_token: Mapping[str, float] | None = None,
    manager: Any | None = None,
) -> Callable[[], dict[str, Any] | None]:
    from config import manipulate_prefs as mp
    from research.strategy_math.strategy_tracker import (
        entry_qty_lots,
        format_entry_legs_copy,
        format_entry_pnl_summary,
        legs_for_projection,
    )

    def provider() -> dict[str, Any] | None:
        stored = entry.get("legs") or []
        if not stored:
            return None
        index = str(entry.get("index") or "NIFTY").upper()
        lot_size = mp.lot_size_for_index(index)
        qty_lots = entry_qty_lots(entry, lot_size=lot_size)
        strategy_key = str(entry.get("strategy_key") or "") or None
        wing_steps = int(entry.get("wing_steps") or 0)
        # Expiry projection uses locked entry fills, not live LTPs (Fix A).
        legs = legs_for_projection(entry, lot_size=lot_size)
        spot = make_index_spot_provider(order_panel, index)()
        if spot is None:
            spot = entry.get("spot_at_entry")
        if spot is None:
            return None
        center_strike = _resolve_center_strike(
            legs,
            strategy_key=strategy_key,
            center_strike=None,
        )
        max_profit = entry.get("max_profit")
        max_risk = entry.get("max_risk")
        lower_be: float | None = None
        upper_be: float | None = None
        if strategy_key:
            from api.multi_leg_margin import estimate_strategy_breakevens

            lower_be, upper_be = estimate_strategy_breakevens(
                strategy_key,
                legs,
                index=index,
                wing_steps=wing_steps,
            )
        unreal = None
        if manager is not None and str(entry.get("status") or "").upper() == "OPEN":
            try:
                unreal = manager.strategy_entry_unrealized(entry)
            except Exception:
                unreal = None
        from research.strategy_math.strategy_ltp_feed import get_strategy_ltp_cache

        ltp_cache = get_strategy_ltp_cache()
        summary = build_projection_summary(
            legs=legs,
            entry=entry,
            strategy_key=strategy_key,
            strategy_label=str(entry.get("strategy_label") or "Strategy"),
            spot=float(spot),
            lower_be=lower_be,
            upper_be=upper_be,
            max_profit_hint=max_profit,
            max_risk_hint=max_risk,
            lot_size=lot_size,
            qty_lots=qty_lots,
            unrealized=unreal,
            index=index,
            wing_steps=wing_steps,
        )
        return {
            "strategy_label": str(entry.get("strategy_label") or "Strategy"),
            "legs": legs,
            "spot": float(spot),
            "index": index,
            "lot_size": lot_size,
            "strategy_key": strategy_key,
            "wing_steps": wing_steps,
            "lower_be": lower_be,
            "upper_be": upper_be,
            "center_strike": center_strike,
            "max_profit_hint": max_profit,
            "max_risk_hint": max_risk,
            "summary": summary,
            "legs_detail": format_entry_legs_copy(entry),
            "pnl_detail": format_entry_pnl_summary(entry, unrealized=unreal),
        }

    return provider


def show_strategy_pnl_projection_from_entry(
    master: tk.Misc,
    entry: Mapping[str, Any],
    *,
    order_panel: Any | None = None,
    unrealized: float | None = None,
    ltp_by_token: Mapping[str, float] | None = None,
    manager: Any | None = None,
) -> None:
    """Open P&L projection for a recorded strategy tracker row."""
    from config import manipulate_prefs as mp
    from research.strategy_math.strategy_tracker import (
        format_entry_legs_copy,
        format_entry_pnl_summary,
        legs_for_projection,
    )

    legs = entry.get("legs") or []
    if not legs:
        raise ValueError("strategy entry has no legs")
    index = str(entry.get("index") or "NIFTY").upper()
    lot_size = mp.lot_size_for_index(index)
    proj_legs = legs_for_projection(entry, lot_size=lot_size)
    spot = entry.get("spot_at_entry")
    if spot is None:
        for leg in legs:
            try:
                spot = float(leg.get("strike"))
                break
            except (TypeError, ValueError):
                continue
    if spot is None:
        raise ValueError("spot unavailable")
    strategy_key = str(entry.get("strategy_key") or "") or None
    wing_steps = int(entry.get("wing_steps") or 0)
    center_strike = _resolve_center_strike(
        proj_legs,
        strategy_key=strategy_key,
        center_strike=None,
    )
    lower_be, upper_be = _resolve_breakevens(
        proj_legs,
        strategy_key=strategy_key,
        index=index,
        wing_steps=wing_steps,
        lower_be=None,
        upper_be=None,
    )
    max_profit_hint: float | None = None
    max_risk_hint: float | None = None
    try:
        raw_mp = entry.get("max_profit")
        if raw_mp is not None:
            max_profit_hint = float(raw_mp)
    except (TypeError, ValueError):
        pass
    try:
        raw_mr = entry.get("max_risk")
        if raw_mr is not None:
            max_risk_hint = float(raw_mr)
    except (TypeError, ValueError):
        pass
    live_spot = make_index_spot_provider(order_panel, index)()
    if live_spot is not None:
        spot = live_spot
    from research.strategy_math.strategy_tracker import entry_qty_lots

    summary = build_projection_summary(
        legs=proj_legs,
        entry=entry,
        strategy_key=strategy_key,
        strategy_label=str(entry.get("strategy_label") or "Strategy"),
        spot=float(spot),
        lower_be=lower_be,
        upper_be=upper_be,
        max_profit_hint=max_profit_hint,
        max_risk_hint=max_risk_hint,
        lot_size=lot_size,
        qty_lots=entry_qty_lots(entry, lot_size=lot_size),
        unrealized=unrealized,
        index=index,
        wing_steps=wing_steps,
    )
    show_strategy_pnl_projection(
        master,
        strategy_label=str(entry.get("strategy_label") or "Strategy"),
        legs=proj_legs,
        spot=float(spot),
        index=index,
        lot_size=lot_size,
        strategy_key=strategy_key,
        wing_steps=wing_steps,
        lower_be=lower_be,
        upper_be=upper_be,
        center_strike=center_strike,
        max_profit_hint=max_profit_hint,
        max_risk_hint=max_risk_hint,
        spot_provider=make_index_spot_provider(order_panel, index),
        live_refresh_provider=make_entry_live_refresh_provider(
            entry,
            order_panel=order_panel,
            ltp_by_token=ltp_by_token,
            manager=manager,
        ),
        legs_detail=format_entry_legs_copy(entry),
        pnl_detail=format_entry_pnl_summary(entry, unrealized=unrealized),
        summary=summary,
        order_panel=order_panel,
        expiry_raw=str(entry.get("expiry_raw") or ""),
    )


def refresh_strategy_pnl_projection_if_open(
    master: tk.Misc,
    *,
    strategy_label: str,
    legs: Sequence[Mapping[str, Any]],
    spot: float | None = None,
    index: str | None = None,
    lot_size: int | None = None,
    half_range: float = 500,
    step: float | None = None,
    strategy_key: str | None = None,
    wing_steps: int = 0,
    lower_be: float | None = None,
    upper_be: float | None = None,
    center_strike: float | None = None,
) -> bool:
    """Update the open projection popup; return True if refresh was scheduled."""
    if not pnl_projection_is_open():
        return False
    global _PROJECTION_WIN
    if spot is None:
        for leg in legs:
            try:
                spot = float(leg.get("strike"))
                break
            except (TypeError, ValueError):
                continue
    if spot is None or not legs:
        return False
    try:
        _PROJECTION_WIN.refresh(
            strategy_label=strategy_label,
            legs=legs,
            spot=float(spot),
            index=index,
            lot_size=lot_size,
            half_range=half_range,
            step=step,
            strategy_key=strategy_key,
            wing_steps=wing_steps,
            lower_be=lower_be,
            upper_be=upper_be,
            center_strike=center_strike,
            reposition=False,
            background=True,
        )
    except tk.TclError:
        _PROJECTION_WIN = None
        return False
    return True

