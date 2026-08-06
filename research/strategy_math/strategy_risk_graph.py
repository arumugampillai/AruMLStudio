"""Tk payoff-at-expiry graph for the strategy panel."""
from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from typing import Any, Mapping, Sequence

from research.strategy_math.strategy_payoff import sample_payoff_curve

_GRAPH_WIN: "StrategyRiskGraphWindow | None" = None

_CLR_SPOT = "#ea580c"
_CLR_CENTER = "#7c3aed"
_CLR_BE = "#16a34a"
_CLR_ZONE_FILL = "#dcfce7"
_CLR_ZONE_EDGE = "#86efac"
_CLR_MUTED = "#64748b"
_CLR_BODY = "#475569"

_HDR_ROW_GAP = 20


def _place_beside_main(
    main_win: tk.Misc,
    child_win: tk.Toplevel,
    *,
    gap: int = 8,
    y_offset: int = 0,
) -> None:
    """Place graph to the right of the main app (or left if no room)."""
    try:
        main_win.update_idletasks()
        child_win.update_idletasks()
    except tk.TclError:
        return
    try:
        x_main = int(main_win.winfo_rootx())
        y_main = int(main_win.winfo_rooty())
        w_main = int(main_win.winfo_width())
        h_main = int(main_win.winfo_height())
        w_child = int(child_win.winfo_width())
        h_child = int(child_win.winfo_height())
    except (tk.TclError, ValueError, TypeError):
        return
    if w_main < 50 or h_main < 50:
        return
    if w_child < 50:
        w_child = 480
    if h_child < 50:
        h_child = 400
    x = x_main + w_main + gap
    y = y_main + int(y_offset)
    try:
        screen_w = int(main_win.winfo_screenwidth())
        screen_h = int(main_win.winfo_screenheight())
    except (tk.TclError, ValueError, TypeError):
        screen_w, screen_h = 1920, 1080
    margin = 12
    if x + w_child > screen_w - margin:
        x = x_main - w_child - gap
    if x < margin:
        x = margin
    if x + w_child > screen_w - margin:
        x = max(margin, screen_w - w_child - margin)
    if y < margin:
        y = margin
    if y + h_child > screen_h - margin:
        y = max(margin, screen_h - h_child - margin)
    try:
        child_win.geometry(f"{w_child}x{h_child}+{x}+{y}")
    except tk.TclError:
        pass


def _fmt_strike(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{int(round(float(value))):,}"


def _fmt_pts(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{int(round(float(value))):,} pts"


def _fmt_be_range(lower_be: float | None, upper_be: float | None) -> str | None:
    if lower_be is not None and upper_be is not None:
        return f"({_fmt_strike(lower_be)}-{_fmt_strike(upper_be)})"
    if lower_be is not None:
        return f"({_fmt_strike(lower_be)})"
    if upper_be is not None:
        return f"({_fmt_strike(upper_be)})"
    return None


class StrategyRiskGraphWindow(tk.Toplevel):
    _CW = 440
    _CH = 280
    _ML = 14
    _MR = 14
    _MT = 12
    _MB = 28
    _FONT_NUM = ("Arial", 9, "bold")
    _HDR_MAX_W = 420

    def __init__(
        self,
        master: tk.Misc,
        *,
        strategy_label: str,
        strategy_key: str = "",
        legs: Sequence[Mapping[str, Any]],
        index: str,
        spot: float | None = None,
        lower_be: float | None = None,
        upper_be: float | None = None,
        max_risk: float | None = None,
        max_profit: float | None = None,
        risk_uncapped: bool = False,
        profit_uncapped: bool = False,
        wing_steps: int = 0,
    ) -> None:
        super().__init__(master)
        self.title(f"Risk graph — {strategy_label}")
        self.resizable(False, False)
        self._anchor_win = master

        tk.Label(
            self,
            text=strategy_label,
            font=("Arial", 10, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(6, 0))

        center, zone_bounds = self._resolve_zone_metrics(
            strategy_key, legs, index=index, wing_steps=wing_steps
        )
        zone_width_pts = self._zone_width_pts(zone_bounds, lower_be, upper_be)

        hdr = tk.Frame(self)
        hdr.pack(fill=tk.X, padx=8, pady=(4, 6))
        self._build_metrics_header(
            hdr,
            strategy_key=strategy_key,
            legs=legs,
            spot=spot,
            lower_be=lower_be,
            upper_be=upper_be,
            max_risk=max_risk,
            max_profit=max_profit,
            risk_uncapped=risk_uncapped,
            profit_uncapped=profit_uncapped,
            center=center,
            zone_width_pts=zone_width_pts,
        )

        self._canvas = tk.Canvas(
            self,
            width=self._CW,
            height=self._CH,
            bg="#fafafa",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
        )
        self._canvas.pack(padx=8, pady=(0, 8))

        self._draw(
            legs=legs,
            index=index,
            spot=spot,
            lower_be=lower_be,
            upper_be=upper_be,
            center=center,
            zone_bounds=zone_bounds,
        )

        tk.Button(self, text="Close", command=self.destroy, font=("Arial", 9)).pack(
            pady=(0, 8)
        )
        _place_beside_main(self._anchor_win, self)
        try:
            self.lift()
        except tk.TclError:
            pass

    @staticmethod
    def _resolve_zone_metrics(
        strategy_key: str,
        legs: Sequence[Mapping[str, Any]],
        *,
        index: str,
        wing_steps: int,
    ) -> tuple[float | None, tuple[float, float, str] | None]:
        from api.multi_leg_margin import (
            estimate_strategy_center,
            estimate_strategy_profit_zone_bounds,
            strategy_shows_center_profit_zone,
        )

        if not strategy_key or not strategy_shows_center_profit_zone(strategy_key):
            return None, None
        center = estimate_strategy_center(strategy_key, legs)
        zone_bounds = estimate_strategy_profit_zone_bounds(
            strategy_key,
            legs,
            index=index,
            wing_steps=wing_steps,
        )
        return center, zone_bounds

    @staticmethod
    def _zone_width_pts(
        zone_bounds: tuple[float, float, str] | None,
        lower_be: float | None,
        upper_be: float | None,
    ) -> float | None:
        if zone_bounds is not None:
            z_lo, z_hi, _mode = zone_bounds
            return float(z_hi) - float(z_lo)
        if lower_be is not None and upper_be is not None:
            return float(upper_be) - float(lower_be)
        return None

    def _text_width(self, text: str, font: tkfont.Font) -> int:
        try:
            return int(font.measure(text)) + 6
        except tk.TclError:
            return len(text) * 7 + 6

    def _pack_header_items(
        self,
        parent: tk.Misc,
        items: list[tuple[str, str]],
        *,
        font: tuple[str, int, str] | tuple[str, int],
    ) -> None:
        if not items:
            return
        tk_font = tkfont.Font(font=font)
        row = tk.Frame(parent)
        row.pack(fill=tk.X, pady=(0, 2))
        x_used = 0
        for text, color in items:
            w = self._text_width(text, tk_font)
            if x_used > 0 and x_used + _HDR_ROW_GAP + w > self._HDR_MAX_W:
                row = tk.Frame(parent)
                row.pack(fill=tk.X, pady=(0, 2))
                x_used = 0
            pad = (0, 0) if x_used == 0 else (_HDR_ROW_GAP, 0)
            tk.Label(row, text=text, font=font, fg=color, anchor="w").pack(
                side=tk.LEFT, padx=pad
            )
            x_used += w + (0 if x_used == 0 else _HDR_ROW_GAP)

    def _build_metrics_header(
        self,
        parent: tk.Misc,
        *,
        strategy_key: str = "",
        legs: Sequence[Mapping[str, Any]] | None = None,
        spot: float | None,
        lower_be: float | None,
        upper_be: float | None,
        max_risk: float | None,
        max_profit: float | None,
        risk_uncapped: bool = False,
        profit_uncapped: bool = False,
        center: float | None,
        zone_width_pts: float | None,
    ) -> None:
        from api.multi_leg_margin import (
            breakeven_distance_from_spot,
            format_strategy_max_risk_display,
        )

        font = self._FONT_NUM

        row1: list[tuple[str, str]] = []
        if spot is not None:
            row1.append((f"Spot: {_fmt_strike(spot)}", _CLR_SPOT))
        risk_txt = format_strategy_max_risk_display(
            strategy_key,
            max_risk,
            uncapped=risk_uncapped,
            legs=legs,
        )
        if risk_txt != "—":
            row1.append((f"Risk: {risk_txt}", _CLR_BODY))
        if profit_uncapped:
            row1.append(("Max Profit: Uncapped", _CLR_BODY))
        elif max_profit is not None:
            row1.append((f"Max Profit: ₹{max_profit:,.0f}", _CLR_BODY))
        self._pack_header_items(parent, row1, font=font)

        row2: list[tuple[str, str]] = []
        be_range = _fmt_be_range(lower_be, upper_be)
        if be_range is not None:
            row2.append((f"BE: {be_range}", _CLR_BE))
        if zone_width_pts is not None:
            row2.append((f"Profit Zone Width: {_fmt_pts(zone_width_pts)}", _CLR_BE))
        self._pack_header_items(parent, row2, font=font)

        row3: list[tuple[str, str]] = []
        if center is not None:
            row3.append((f"Center: {_fmt_strike(center)}", _CLR_CENTER))
        if spot is not None and lower_be is not None:
            arrow, pts = breakeven_distance_from_spot(spot, lower_be)
            row3.append((f"{arrow} To Lower BE: {pts:,} pts", _CLR_BODY))
        if spot is not None and upper_be is not None:
            arrow, pts = breakeven_distance_from_spot(spot, upper_be)
            row3.append((f"{arrow} To Upper BE: {pts:,} pts", _CLR_BODY))
        self._pack_header_items(parent, row3, font=font)

    def _plot_rect(self) -> tuple[int, int, int, int]:
        return (
            self._ML,
            self._MT,
            self._CW - self._MR,
            self._CH - self._MB,
        )

    def _draw(
        self,
        *,
        legs: Sequence[Mapping[str, Any]],
        index: str,
        spot: float | None,
        lower_be: float | None,
        upper_be: float | None,
        center: float | None,
        zone_bounds: tuple[float, float, str] | None,
    ) -> None:
        c = self._canvas
        c.delete("all")
        x0, y0, x1, y1 = self._plot_rect()

        curve = sample_payoff_curve(legs, index=index, spot=spot)
        if not curve:
            c.create_text(
                self._CW // 2,
                self._CH // 2,
                text="No payoff data",
                fill=_CLR_MUTED,
            )
            return

        spots = [p for p, _ in curve]
        pnls = [v for _, v in curve]
        xmin, xmax = min(spots), max(spots)
        for edge in (lower_be, upper_be, spot, center):
            if edge is None:
                continue
            try:
                ev = float(edge)
            except (TypeError, ValueError):
                continue
            xmin = min(xmin, ev)
            xmax = max(xmax, ev)
        if zone_bounds is not None:
            z_lo, z_hi, _ = zone_bounds
            xmin = min(xmin, float(z_lo), float(z_hi))
            xmax = max(xmax, float(z_lo), float(z_hi))

        ymin, ymax = min(pnls), max(pnls)
        if abs(ymax - ymin) < 1e-6:
            ymax = ymin + 1.0
        pad_y = max((ymax - ymin) * 0.08, 1.0)
        ymin -= pad_y
        ymax += pad_y
        if ymin < 0 < ymax:
            pass
        elif ymax <= 0:
            ymax = pad_y
        elif ymin >= 0:
            ymin = -pad_y

        def sx(spot_v: float) -> float:
            if xmax <= xmin:
                return (x0 + x1) / 2
            return x0 + (spot_v - xmin) / (xmax - xmin) * (x1 - x0)

        def sy(pnl: float) -> float:
            return y1 - (pnl - ymin) / (ymax - ymin) * (y1 - y0)

        # profit zone shading only (no text)
        if zone_bounds is not None:
            z_lo, z_hi, z_mode = zone_bounds
            if z_mode == "inner":
                zx0 = max(x0, min(sx(z_lo), sx(z_hi)))
                zx1 = min(x1, max(sx(z_lo), sx(z_hi)))
                if zx1 > zx0:
                    c.create_rectangle(
                        zx0,
                        y0,
                        zx1,
                        y1,
                        fill=_CLR_ZONE_FILL,
                        outline=_CLR_ZONE_EDGE,
                        width=1,
                    )
            elif z_mode == "outer":
                lx1 = max(x0, min(sx(z_lo), x1))
                rx0 = min(x1, max(sx(z_hi), x0))
                if lx1 > x0:
                    c.create_rectangle(
                        x0, y0, lx1, y1, fill=_CLR_ZONE_FILL, outline=_CLR_ZONE_EDGE, width=1
                    )
                if rx0 < x1:
                    c.create_rectangle(
                        rx0, y0, x1, y1, fill=_CLR_ZONE_FILL, outline=_CLR_ZONE_EDGE, width=1
                    )

        # plot frame + zero line
        c.create_rectangle(x0, y0, x1, y1, outline="#cbd5e1")
        zy = sy(0.0)
        if y0 <= zy <= y1:
            c.create_line(x0, zy, x1, zy, fill="#94a3b8", dash=(4, 3))

        # breakeven lines (green dashed, secondary)
        for be in (lower_be, upper_be):
            if be is None:
                continue
            try:
                bx = sx(float(be))
            except (TypeError, ValueError):
                continue
            if x0 <= bx <= x1:
                c.create_line(bx, y0, bx, y1, fill=_CLR_BE, dash=(4, 3), width=1)

        # center line (purple, thinner secondary)
        if center is not None:
            try:
                cx = sx(float(center))
            except (TypeError, ValueError):
                cx = None
            if cx is not None and x0 <= cx <= x1:
                c.create_line(cx, y0, cx, y1, fill=_CLR_CENTER, width=1)

        # payoff curve
        pts: list[float] = []
        for spot_v, pnl in curve:
            pts.extend((sx(spot_v), sy(pnl)))
        c.create_line(*pts, fill="#1d4ed8", width=2, smooth=False)

        # spot line (orange, primary — drawn last, thicker)
        if spot is not None:
            try:
                spx = sx(float(spot))
            except (TypeError, ValueError):
                spx = None
            if spx is not None and x0 <= spx <= x1:
                c.create_line(spx, y0, spx, y1, fill=_CLR_SPOT, width=2)


def show_strategy_risk_graph(
    master: tk.Misc,
    *,
    strategy_label: str,
    strategy_key: str = "",
    legs: Sequence[Mapping[str, Any]],
    index: str,
    spot: float | None = None,
    lower_be: float | None = None,
    upper_be: float | None = None,
    max_risk: float | None = None,
    max_profit: float | None = None,
    risk_uncapped: bool = False,
    profit_uncapped: bool = False,
    wing_steps: int = 0,
) -> None:
    global _GRAPH_WIN
    try:
        if _GRAPH_WIN is not None and _GRAPH_WIN.winfo_exists():
            _GRAPH_WIN.destroy()
    except tk.TclError:
        pass
    _GRAPH_WIN = StrategyRiskGraphWindow(
        master,
        strategy_label=strategy_label,
        strategy_key=strategy_key,
        legs=legs,
        index=index,
        spot=spot,
        lower_be=lower_be,
        upper_be=upper_be,
        max_risk=max_risk,
        max_profit=max_profit,
        risk_uncapped=risk_uncapped,
        profit_uncapped=profit_uncapped,
        wing_steps=wing_steps,
    )
