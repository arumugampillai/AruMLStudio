"""Canvas chart helpers for Fold Research panel."""

from __future__ import annotations

import math
import tkinter as tk
from typing import Any


def draw_line_chart(
    canvas: tk.Canvas,
    points: list[float] | None = None,
    *,
    series: list[tuple[str, list[float], str]] | None = None,
    secondary_series: list[tuple[str, list[float], str]] | None = None,
    title: str = "",
    color: str = "#58a6ff",
    fill: str = "#1a2a44",
    pad: int = 28,
    empty_message: str = "No data",
    include_zero: bool = True,
    cursor_index: int | None = None,
) -> dict[str, Any]:
    """Draw one or more line series on a canvas.

    Backward compatible: pass ``points`` for a single series.
    Multi-series: ``series=[(label, ys, color), ...]`` with shared x-index
    and a union y-range. Optional ``secondary_series`` uses a right-hand y-scale
    (dual-axis overlay). Optional ``cursor_index`` draws a vertical crosshair.

    Returns a layout dict for hit-testing (``pad``, ``inner_w``, ``n_points``, …).
    """
    canvas.delete("all")
    w = max(canvas.winfo_width(), 200)
    h = max(canvas.winfo_height(), 120)
    canvas.create_rectangle(0, 0, w, h, fill=fill, outline="")
    empty_layout: dict[str, Any] = {
        "pad": pad,
        "top_pad": pad,
        "inner_w": max(w - pad * 2, 1),
        "inner_h": max(h - pad * 2, 1),
        "n_points": 0,
        "w": w,
        "h": h,
    }
    if title:
        canvas.create_text(pad, 12, text=title, anchor="w", fill="#aab", font=("Segoe UI", 9, "bold"))

    drawn: list[tuple[str, list[float], str]] = []
    if series:
        drawn = [(str(lab), list(ys), str(col)) for lab, ys, col in series if ys]
    elif points:
        drawn = [("", list(points), color)]

    secondary: list[tuple[str, list[float], str]] = []
    if secondary_series:
        secondary = [
            (str(lab), list(ys), str(col)) for lab, ys, col in secondary_series if ys
        ]

    if not drawn and not secondary:
        canvas.create_text(w / 2, h / 2, text=empty_message, fill="#666", font=("Segoe UI", 9))
        return empty_layout

    def _finite_vals(ys: list[float]) -> list[float]:
        out: list[float] = []
        for v in ys:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                out.append(fv)
        return out

    def _y_range(series_list: list[tuple[str, list[float], str]], *, force_zero: bool) -> tuple[float, float] | None:
        all_vals: list[float] = []
        for _lab, ys, _col in series_list:
            seq = list(ys)
            if len(seq) == 1:
                seq = [seq[0], seq[0]]
            all_vals.extend(_finite_vals(seq))
        if not all_vals:
            return None
        lo = min(all_vals)
        hi = max(all_vals)
        if force_zero:
            lo = min(lo, 0.0)
            hi = max(hi, 0.0)
        if hi == lo:
            hi = lo + 1.0
        return lo, hi

    primary_range = _y_range(drawn, force_zero=include_zero) if drawn else None
    secondary_range = _y_range(secondary, force_zero=False) if secondary else None
    if primary_range is None and secondary_range is None:
        canvas.create_text(w / 2, h / 2, text=empty_message, fill="#666", font=("Segoe UI", 9))
        return empty_layout
    if primary_range is None and secondary_range is not None:
        # Promote secondary to primary when only right-axis series given.
        drawn, secondary = secondary, []
        primary_range, secondary_range = secondary_range, None

    assert primary_range is not None
    lo, hi = primary_range
    sec_lo, sec_hi = secondary_range if secondary_range else (0.0, 1.0)

    legend_items = [(lab, col) for lab, _ys, col in drawn if lab]
    legend_items.extend((lab, col) for lab, _ys, col in secondary if lab)
    legend_h = 14 if legend_items else 0
    right_pad = pad + (36 if secondary else 0)
    top_pad = pad + (4 if title else 0) + legend_h
    inner_w = w - pad - right_pad
    inner_h = h - top_pad - pad
    if inner_w < 1:
        inner_w = 1
    if inner_h < 1:
        inner_h = 1

    n_points = 0
    for _lab, ys, _col in drawn + secondary:
        n_points = max(n_points, len(ys))

    layout: dict[str, Any] = {
        "pad": pad,
        "top_pad": top_pad,
        "inner_w": inner_w,
        "inner_h": inner_h,
        "n_points": n_points,
        "w": w,
        "h": h,
        "right_pad": right_pad,
    }

    if include_zero and lo <= 0.0 <= hi:
        zero_y = top_pad + inner_h * (hi - 0.0) / (hi - lo)
        canvas.create_line(pad, zero_y, w - right_pad, zero_y, fill="#334", dash=(3, 3))

    if legend_h:
        lx = pad
        ly = 22 if title else 10
        for lab, col in legend_items:
            canvas.create_rectangle(lx, ly - 4, lx + 10, ly + 4, fill=col, outline="")
            canvas.create_text(lx + 14, ly, text=lab, anchor="w", fill="#99a", font=("Segoe UI", 8))
            lx += 14 + max(40, len(lab) * 7) + 10

    if secondary:
        canvas.create_text(
            pad, h - 8, text=f"{lo:.3g}…{hi:.3g}", anchor="w", fill="#667", font=("Segoe UI", 7)
        )
        canvas.create_text(
            w - 8,
            h - 8,
            text=f"{sec_lo:.3g}…{sec_hi:.3g}",
            anchor="e",
            fill="#667",
            font=("Segoe UI", 7),
        )

    def _draw_series(
        series_list: list[tuple[str, list[float], str]],
        y_lo: float,
        y_hi: float,
    ) -> None:
        for _lab, ys, col in series_list:
            seq = list(ys)
            if len(seq) == 1:
                seq = [seq[0], seq[0]]
            n = max(len(seq) - 1, 1)
            segment: list[float] = []

            def _flush() -> None:
                nonlocal segment
                if len(segment) >= 4:
                    canvas.create_line(*segment, fill=col, width=2, smooth=True)
                segment = []

            for i, raw in enumerate(seq):
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    _flush()
                    continue
                if not math.isfinite(v):
                    _flush()
                    continue
                x = pad + (inner_w * i / n)
                y = top_pad + inner_h * (y_hi - v) / (y_hi - y_lo)
                segment.extend([x, y])
            _flush()

    _draw_series(drawn, lo, hi)
    if secondary:
        _draw_series(secondary, sec_lo, sec_hi)

    if cursor_index is not None and n_points > 0:
        ci = max(0, min(int(cursor_index), n_points - 1))
        n_seg = max(n_points - 1, 1)
        cx = pad + (inner_w * ci / n_seg) if n_points > 1 else pad + inner_w / 2
        canvas.create_line(
            cx, top_pad, cx, top_pad + inner_h, fill="#e6edf3", width=1, dash=(4, 3), tags="crosshair"
        )

    try:
        canvas._chart_layout = layout  # type: ignore[attr-defined]
    except Exception:
        pass
    return layout


def draw_histogram(
    canvas: tk.Canvas,
    values: list[float],
    *,
    title: str = "",
    color: str = "#4caf50",
    fill: str = "#1a2a44",
    bins: int = 12,
    pad: int = 28,
) -> None:
    canvas.delete("all")
    w = max(canvas.winfo_width(), 200)
    h = max(canvas.winfo_height(), 120)
    canvas.create_rectangle(0, 0, w, h, fill=fill, outline="")
    if title:
        canvas.create_text(pad, 12, text=title, anchor="w", fill="#aab", font=("Segoe UI", 9, "bold"))
    if not values:
        canvas.create_text(w / 2, h / 2, text="No data", fill="#666", font=("Segoe UI", 9))
        return
    lo, hi = min(values), max(values)
    if lo == hi:
        counts = [len(values)]
    else:
        width = (hi - lo) / bins
        counts = [0] * bins
        for v in values:
            idx = min(bins - 1, int((v - lo) / width) if width else 0)
            counts[idx] += 1
    max_count = max(counts) or 1
    inner_w = w - pad * 2
    inner_h = h - pad * 2
    bar_w = inner_w / len(counts)
    for i, count in enumerate(counts):
        bh = inner_h * count / max_count
        x0 = pad + i * bar_w + 1
        x1 = pad + (i + 1) * bar_w - 1
        y0 = pad + inner_h - bh
        y1 = pad + inner_h
        canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")


def draw_bucket_bars(
    canvas: tk.Canvas,
    buckets: list[tuple[str, int]],
    *,
    title: str = "",
    color: str = "#58a6ff",
    fill: str = "#1a2a44",
    pad: int = 28,
) -> None:
    """Fixed-label histogram: buckets = [(label, count), ...]."""
    canvas.delete("all")
    w = max(canvas.winfo_width(), 200)
    h = max(canvas.winfo_height(), 140)
    canvas.create_rectangle(0, 0, w, h, fill=fill, outline="")
    if title:
        canvas.create_text(pad, 12, text=title, anchor="w", fill="#aab", font=("Segoe UI", 9, "bold"))
    if not buckets:
        canvas.create_text(w / 2, h / 2, text="No data", fill="#666", font=("Segoe UI", 9))
        return
    max_count = max(c for _, c in buckets) or 1
    inner_w = w - pad * 2
    inner_h = h - pad * 2 - 16
    bar_w = inner_w / len(buckets)
    for i, (label, count) in enumerate(buckets):
        bh = inner_h * count / max_count
        x0 = pad + i * bar_w + 2
        x1 = pad + (i + 1) * bar_w - 2
        y0 = pad + inner_h - bh
        y1 = pad + inner_h
        canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
        canvas.create_text(
            (x0 + x1) / 2,
            y1 + 10,
            text=label,
            fill="#888",
            font=("Segoe UI", 8),
        )
        if count > 0:
            canvas.create_text(
                (x0 + x1) / 2,
                y0 - 6,
                text=str(count),
                fill="#ccc",
                font=("Segoe UI", 8),
            )


def fmt_ts(ts: Any) -> str:
    try:
        return f"{float(ts):.1f}s"
    except (TypeError, ValueError):
        return "—"


def draw_sparkline(
    canvas: tk.Canvas,
    points: list[float],
    *,
    title: str = "",
    color: str = "#1976D2",
    fill: str = "#f8f9fa",
    pad: int = 24,
) -> None:
    """Compact line chart for trade replay (light theme)."""
    canvas.delete("all")
    w = max(canvas.winfo_width(), 120)
    h = max(canvas.winfo_height(), 72)
    canvas.create_rectangle(0, 0, w, h, fill=fill, outline="#ddd")
    if title:
        canvas.create_text(pad, 10, text=title, anchor="w", fill="#444", font=("Segoe UI", 8, "bold"))
    if not points:
        canvas.create_text(w / 2, h / 2, text="No data", fill="#888", font=("Segoe UI", 8))
        return
    if len(points) == 1:
        points = [points[0], points[0]]
    lo, hi = min(points), max(points)
    if hi == lo:
        hi = lo + 0.01
    inner_w = w - pad * 2
    inner_h = h - pad * 2
    coords: list[float] = []
    for i, v in enumerate(points):
        x = pad + (inner_w * i / (len(points) - 1))
        y = pad + inner_h * (hi - v) / (hi - lo)
        coords.extend([x, y])
    canvas.create_line(*coords, fill=color, width=2, smooth=True)
    canvas.create_text(pad, h - 6, text=f"{points[0]:.2f}", anchor="w", fill="#666", font=("Segoe UI", 7))
    canvas.create_text(w - pad, h - 6, text=f"{points[-1]:.2f}", anchor="e", fill="#666", font=("Segoe UI", 7))


def draw_confidence_bar(
    canvas: tk.Canvas,
    pct: float | None,
    *,
    title: str = "Prediction Confidence",
    width_blocks: int = 12,
    fill: str = "#f8f9fa",
) -> None:
    """Visual confidence bar e.g. ████████░░░░ 63% · High."""
    canvas.delete("all")
    w = max(canvas.winfo_width(), 200)
    h = max(canvas.winfo_height(), 36)
    canvas.create_rectangle(0, 0, w, h, fill=fill, outline="#ddd")
    if pct is None:
        canvas.create_text(pad := 8, h / 2, text=f"{title}: —", anchor="w", fill="#888", font=("Segoe UI", 9))
        return
    p = max(0.0, min(100.0, float(pct)))
    filled = int(round(p * width_blocks / 100.0))
    blocks = "█" * filled + "░" * (width_blocks - filled)
    if p >= 85:
        tier, color = "Very High", "#1b5e20"
    elif p >= 70:
        tier, color = "High", "#2e7d32"
    elif p >= 50:
        tier, color = "Medium", "#e65100"
    else:
        tier, color = "Low", "#c62828"
    canvas.create_text(8, 10, text=title, anchor="w", fill="#444", font=("Segoe UI", 8, "bold"))
    canvas.create_text(8, 24, text=f"{blocks}  {p:.0f}%  ·  {tier}", anchor="w", fill=color, font=("Consolas", 10))


def resolve_main_app_root(widget: tk.Misc) -> tk.Misc:
    """Return the primary Tk root (not an intermediate tool Toplevel)."""
    w: tk.Misc | None = widget
    found: tk.Misc = widget.winfo_toplevel()
    while w is not None:
        if isinstance(w, tk.Tk):
            return w
        w = w.master  # type: ignore[assignment]
    return found


def place_toplevel_beside_main(win: tk.Toplevel, master: tk.Misc) -> None:
    """Position a toplevel immediately to the right of the main app, same size."""
    root = resolve_main_app_root(master)
    root.update_idletasks()
    w = max(int(root.winfo_width()), 800)
    h = max(int(root.winfo_height()), 600)
    x = int(root.winfo_x())
    y = int(root.winfo_y())
    win.geometry(f"{w}x{h}+{x + w}+{y}")
    win.minsize(640, 480)


def place_toplevel_over_main(
    win: tk.Toplevel,
    master: tk.Misc,
    *,
    scale: float | None = None,
    width_scale: float = 0.5,
    height_scale: float = 0.5,
) -> None:
    """Center a toplevel over the main app at the given width/height fractions."""
    if scale is not None:
        width_scale = scale
        height_scale = scale
    root = resolve_main_app_root(master)
    root.update_idletasks()
    rw = max(int(root.winfo_width()), 800)
    rh = max(int(root.winfo_height()), 600)
    rx = int(root.winfo_x())
    ry = int(root.winfo_y())
    w = max(int(rw * width_scale), 420)
    h = max(int(rh * height_scale), 320)
    x = rx + max(0, (rw - w) // 2)
    y = ry + max(0, (rh - h) // 2)
    win.geometry(f"{w}x{h}+{x}+{y}")
    win.minsize(min(420, w), min(320, h))


def center_toplevel_on_widget(win: tk.Toplevel, anchor: tk.Misc) -> None:
    """Center a toplevel over a widget (e.g. Feature Transformations panel)."""
    win.update_idletasks()
    try:
        anchor.update_idletasks()
    except tk.TclError:
        pass
    aw = max(int(anchor.winfo_width()), 1)
    ah = max(int(anchor.winfo_height()), 1)
    ax = int(anchor.winfo_rootx())
    ay = int(anchor.winfo_rooty())
    ww = max(int(win.winfo_width()), int(win.winfo_reqwidth()), 1)
    wh = max(int(win.winfo_height()), int(win.winfo_reqheight()), 1)
    x = ax + max(0, (aw - ww) // 2)
    y = ay + max(0, (ah - wh) // 2)
    win.geometry(f"+{x}+{y}")
